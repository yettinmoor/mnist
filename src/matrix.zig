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

        // // Square matrix functions
        // pub usingnamespace if (m != n) struct {} else struct {
        //     pub fn identity() Self {
        //         comptime var res = Self.zeroes();
        //         inline for (res.data) |*v, i| {
        //             const r = i / n;
        //             const c = i % n;
        //             if (r == c) {
        //                 v.* = 1;
        //             }
        //         }
        //         return res;
        //     }
        // };

        // Vector functions
        // pub usingnamespace if (n != 1) struct {} else struct {
        // pub fn dot(self: Self, other: Self) T {
        //     var res: T = 0;
        //     for (self.data) |x, i| res += x * other.data[i];
        //     return res;
        // }

        // pub fn mag(self: Self) T {
        //     return math.sqrt(self.dot(self));
        // }

        // pub fn normalize(self: Self) Self {
        //     return self.mulScalar(1 / self.mag());
        // }

        // pub fn reduce(self: Self) Vector(T, m - 1) {
        //     return Vector(T, m - 1).fromSlice(self.data[0 .. m - 1]);
        // }

        // // Vec3 cross product
        // pub usingnamespace if (m != 3) struct {} else struct {
        //     pub fn cross(self: Self, other: Self) Self {
        //         const ax = self.data[0];
        //         const ay = self.data[1];
        //         const az = self.data[2];
        //         const bx = other.data[0];
        //         const by = other.data[1];
        //         const bz = other.data[2];
        //         return .{
        //             .data = [_]T{
        //                 ay * bz - az * by,
        //                 az * bx - ax * bz,
        //                 ax * by - ay * bx,
        //             },
        //         };
        //     }
        // };

        // // new(...)
        // pub usingnamespace switch (m) {
        //     3 => struct {
        //         pub fn new(x: T, y: T, z: T) Self {
        //             return .{ .data = [_]T{ x, y, z } };
        //         }
        //     },
        //     4 => struct {
        //         pub fn new(x: T, y: T, z: T, w: T) Self {
        //             return .{ .data = [_]T{ x, y, z, w } };
        //         }
        //     },
        //     else => struct {},
        // };
        // };

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

            pub fn format(value: Size, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
                try writer.print("{}x{}", .{ value.m, value.n });
            }
        };
    };
}

const testing = std.testing;
const talloc = testing.allocator;

test "add, sub" {
    var m = try Matrix(i32).init(talloc, 2, 2);
    mem.copy(i32, m.data, &[_]i32{ 1, 2, 3, 4 });
    var n = try Matrix(i32).init(talloc, 2, 2);
    mem.copy(i32, n.data, &[_]i32{ 5, 6, 7, 8 });

    var p = try m.clone(talloc);
    p.add(n);
    testing.expectEqualSlices(i32, &[_]i32{ 6, 8, 10, 12 }, p.data);

    var q = try n.clone(talloc);
    q.sub(m);
    testing.expectEqualSlices(i32, &[_]i32{ 4, 4, 4, 4 }, q.data);

    m.deinit();
    n.deinit();
    p.deinit();
    q.deinit();
}

test "matmul square" {
    var m = try Matrix(u32).init(talloc, 2, 2);
    mem.copy(u32, m.data, &[_]u32{ 1, 2, 3, 4 });
    var n = try Matrix(u32).init(talloc, 2, 2);
    mem.copy(u32, n.data, &[_]u32{ 5, 6, 7, 8 });

    var p = try m.mul(n, talloc);
    testing.expectEqualSlices(u32, &[_]u32{ 19, 22, 43, 50 }, p.data);

    m.deinit();
    n.deinit();
    p.deinit();
}

test "matmul nonsquare" {
    var m = try Matrix(u32).init(talloc, 2, 3);
    mem.copy(u32, m.data, &[_]u32{ 1, 2, 3, 4, 5, 6 });
    var n = try Matrix(u32).init(talloc, 3, 2);
    mem.copy(u32, n.data, &[_]u32{ 7, 8, 9, 10, 11, 12 });

    var p = try m.mul(n, talloc);
    testing.expectEqualSlices(u32, &[_]u32{ 58, 64, 139, 154 }, p.data);
    testing.expectEqual(@as(usize, 2), p.m);
    testing.expectEqual(@as(usize, 2), p.n);

    m.deinit();
    n.deinit();
    p.deinit();
}

test "transpose" {
    var m = try Matrix(u32).init(talloc, 2, 3);
    mem.copy(u32, m.data, &[_]u32{ 1, 2, 3, 4, 5, 6 });

    var p = try m.transpose();
    testing.expectEqualSlices(u32, &[_]u32{ 1, 4, 2, 5, 3, 6 }, p.data);
    testing.expectEqual(@as(usize, 3), p.m);
    testing.expectEqual(@as(usize, 2), p.n);

    m.deinit();
    p.deinit();
}
// test "mat * vec" {
//     const m = Matrix(usize, 3, 3){
//         .data = [_]usize{
//             3, 0, 0,
//             0, 2, 0,
//             0, 0, 1,
//         },
//     };
//     const v = Vector(usize, 3).from([_]usize{ 10, 20, 30 });
//     const res = m.mul(v);
//     std.testing.expectEqual([_]usize{ 30, 40, 30 }, res.data);
//     std.testing.expectEqual(Vector(usize, 3), @TypeOf(res));
// }

// test "identity" {
//     const m = Matrix(usize, 3, 3).identity();
//     std.testing.expectEqual([_]usize{
//         1, 0, 0,
//         0, 1, 0,
//         0, 0, 1,
//     }, m.data);
// }

// test "vector dot" {
//     const v = Vector(f32, 3).new(1, 2, 3);
//     const w = Vector(f32, 3).new(6, 5, 4);
//     std.testing.expectEqual(@as(f32, 28), v.dot(w));
// }

// test "vector normalize" {
//     const v = Vector(f32, 3).new(32.2, 123.0, 121.56);
//     const mag = v.normalize().mag();
//     std.testing.expectWithinEpsilon(@as(f32, 1.0), mag, @as(f32, 0.01));
// }

// test "reduce" {
//     const v = Vector(f32, 4){ .data = [_]f32{ 1, 2, 3, 4 } };
//     const w = Vector(f32, 3){ .data = [_]f32{ 1, 2, 3 } };
//     std.testing.expectEqual(w, v.reduce());
// }

// test "cross product" {
//     const v = Vector(f32, 3).new(2, 3, 4);
//     const w = Vector(f32, 3).new(5, 6, 7);
//     std.testing.expectEqual(Vector(f32, 3).new(-3, 6, -3), v.cross(w));
// }
