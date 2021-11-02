const std = @import("std");
const mem = std.mem;
const math = std.math;
const assert = std.debug.assert;

pub fn Matrix(comptime T: type) type {
    return struct {
        allocator: *mem.Allocator,
        data: []T,
        m: u32,
        n: u32,

        const Self = @This();

        pub fn init(allocator: *mem.Allocator, m: usize, n: usize) !Self {
            const data = try allocator.alloc(T, @intCast(u32, m) * @intCast(u32, n));
            return Self{
                .allocator = allocator,
                .data = data,
                .m = @intCast(u32, m),
                .n = @intCast(u32, n),
            };
        }

        pub fn deinit(self: *const Self) void {
            self.allocator.free(self.data);
        }

        pub fn clone(self: *const Self, allocator: *mem.Allocator) !Self {
            const data = try allocator.dupe(T, self.data);
            return Self{
                .allocator = allocator,
                .data = data,
                .m = self.m,
                .n = self.n,
            };
        }

        pub fn size(self: Self) Size {
            return .{ .m = self.m, .n = self.n };
        }

        pub fn copyColumn(self: *Self, xs: []T, column: usize) void {
            for (xs) |x, i| {
                self.data[column + self.n * i] = x;
            }
        }

        pub fn transpose(self: *Self, allocator: *mem.Allocator) !Self {
            var res = try Self.init(allocator, self.n, self.m);
            for (res.data) |*x, i| {
                const r = i / self.m;
                const c = i % self.m;
                x.* = self.data[c * self.n + r];
            }
            return res;
        }

        pub fn add(self: *Self, other: Self) void {
            assert(self.m == other.m and self.n == other.n);
            for (self.data) |*x, i| {
                x.* += other.data[i];
            }
        }

        pub fn sub(self: *Self, other: Self) void {
            assert(self.m == other.m and self.n == other.n);
            for (self.data) |*x, i| {
                x.* -= other.data[i];
            }
        }

        pub fn neg(self: *Self) void {
            for (self.data) |*x| {
                x.* = -x.*;
            }
            return self.opTypeResult(op);
        }

        pub fn mulElem(self: *Self, other: Self) void {
            assert(self.m == other.m and self.n == other.n);
            for (self.data) |*x, i| {
                x.* *= other.data[i];
            }
        }

        pub fn mul(self: Self, other: Self, allocator: *mem.Allocator) !Self {
            assert(self.n == other.m);
            var res = try Self.init(allocator, self.m, other.n);
            for (res.data) |*x, i| {
                const row = i / res.n;
                const col = i % res.n;
                var val: T = 0;
                for (self.data[row * self.n .. (row + 1) * self.n]) |a, j| {
                    const b = other.data[res.n * j + col];
                    val += a * b;
                }
                x.* = val;
            }
            return res;
        }

        pub fn dot(self: Self, other: Self) T {
            assert(self.m * self.n == other.m * other.n);
            var res: T = 0;
            for (self.data) |x, i| {
                res += x * other.data[i];
            }
            return res;
        }

        pub fn mulScalar(self: *Self, scalar: T) void {
            for (self.data) |*x, i| {
                x.* = self.data[i] * scalar;
            }
        }

        pub fn apply(
            self: *const Self,
            allocator: *mem.Allocator,
            comptime R: type,
            comptime func: fn (T) R,
        ) !Self {
            var res = self.*;
            res.data = try allocator.alloc(R, self.data.len);
            for (res.data) |*x, i| {
                x.* = func(self.data[i]);
            }
            return res;
        }

        pub fn format(
            value: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            var cursor: usize = 0;
            const size = value.m * value.n;
            while (cursor < size) : (cursor += value.n) {
                try writer.writeAll("[ ");
                for (value.data[cursor .. cursor + value.n]) |x, i| {
                    try std.fmt.format(writer, "{}", .{x});
                    if (i == value.n - 1) {
                        try writer.writeAll(" ]\n");
                    } else {
                        try writer.writeAll(", ");
                    }
                }
            }
        }

        const Size = struct {
            m: u32,
            n: u32,

            pub fn format(
                value: Size,
                comptime fmt: []const u8,
                options: std.fmt.FormatOptions,
                writer: anytype,
            ) !void {
                try writer.print("{}x{}", .{ value.m, value.n });
            }
        };
    };
}

const testing = std.testing;

test "add, sub" {
    var m = try Matrix(i32).init(testing.allocator, 2, 2);
    mem.copy(i32, m.data, &[_]i32{ 1, 2, 3, 4 });
    var n = try Matrix(i32).init(testing.allocator, 2, 2);
    mem.copy(i32, n.data, &[_]i32{ 5, 6, 7, 8 });

    var p = try m.clone(testing.allocator);
    p.add(n);
    try testing.expectEqualSlices(i32, &[_]i32{ 6, 8, 10, 12 }, p.data);

    var q = try n.clone(testing.allocator);
    q.sub(m);
    try testing.expectEqualSlices(i32, &[_]i32{ 4, 4, 4, 4 }, q.data);

    m.deinit();
    n.deinit();
    p.deinit();
    q.deinit();
}

test "matmul square" {
    var m = try Matrix(u32).init(testing.allocator, 2, 2);
    mem.copy(u32, m.data, &[_]u32{ 1, 2, 3, 4 });
    var n = try Matrix(u32).init(testing.allocator, 2, 2);
    mem.copy(u32, n.data, &[_]u32{ 5, 6, 7, 8 });

    var p = try m.mul(n, testing.allocator);
    try testing.expectEqualSlices(u32, &[_]u32{ 19, 22, 43, 50 }, p.data);

    m.deinit();
    n.deinit();
    p.deinit();
}

test "matmul nonsquare" {
    var m = try Matrix(u32).init(testing.allocator, 2, 3);
    mem.copy(u32, m.data, &[_]u32{ 1, 2, 3, 4, 5, 6 });
    var n = try Matrix(u32).init(testing.allocator, 3, 2);
    mem.copy(u32, n.data, &[_]u32{ 7, 8, 9, 10, 11, 12 });

    var p = try m.mul(n, testing.allocator);
    try testing.expectEqualSlices(u32, &[_]u32{ 58, 64, 139, 154 }, p.data);
    try testing.expectEqual(@as(usize, 2), p.m);
    try testing.expectEqual(@as(usize, 2), p.n);

    m.deinit();
    n.deinit();
    p.deinit();
}

test "transpose" {
    var m = try Matrix(u32).init(testing.allocator, 2, 3);
    mem.copy(u32, m.data, &[_]u32{ 1, 2, 3, 4, 5, 6 });

    var p = try m.transpose(testing.allocator);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 4, 2, 5, 3, 6 }, p.data);
    try testing.expectEqual(@as(usize, 3), p.m);
    try testing.expectEqual(@as(usize, 2), p.n);

    m.deinit();
    p.deinit();
}
