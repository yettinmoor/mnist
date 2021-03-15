const std = @import("std");
const io = std.io;
const fs = std.fs;
const mem = std.mem;
const math = std.math;
const ArrayList = std.ArrayList;
const MultiArrayList = std.MultiArrayList;

const Network = @import("Network.zig");

const matrix = @import("matrix.zig");
const Matrix = matrix.Matrix;
const Vector = matrix.Vector;

const Image = struct {
    label: u8,
    data: [28 * 28]u8,

    pub fn toInput(img: Image, allocator: *mem.Allocator) !Matrix(f32) {
        var input = try Matrix(f32).init(allocator, img.data.len, 1);
        for (img.data) |d, i| {
            input.data[i] = @intToFloat(f32, d) / 255;
        }
        return input;
    }

    pub fn desired(img: Image, allocator: *mem.Allocator) !Matrix(f32) {
        var res = try Matrix(f32).init(allocator, 10, 1);
        for (res.data) |*d, i| {
            d.* = if (img.label == i) 1 else 0;
        }
        return res;
    }

    pub fn cost(img: Image, output: Matrix(f32)) f32 {
        var c: f32 = 0;
        for (output.data) |o, i| {
            c += math.pow(f32, o - if (img.label == i) @as(f32, 1) else @as(f32, 0), 2);
        }
        return c;
    }
};

pub fn main() anyerror!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = &gpa.allocator;
    defer _ = gpa.detectLeaks();

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.os.getrandom(std.mem.asBytes(&seed)) catch unreachable;
        break :blk seed;
    });
    const rand = &prng.random;

    const layers = [_]u32{ 28 * 28, 16, 16, 10 };
    const training_data = try getImages(allocator, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    defer allocator.free(training_data);

    const cwd = fs.cwd();

    // Get arg (weight file)
    const args = try std.process.argsAlloc(allocator);
    defer allocator.free(args);

    var nw = if (args.len > 1) blk: {
        // Deserialize network from file.
        const filename = args[1];
        const file = cwd.openFile(filename, .{}) catch |err| {
            try io.getStdErr().writer().print("Could not find or open file `{s}`.\n", .{filename});
            std.process.exit(1);
        };
        break :blk try Network.deserialize(allocator, file.reader());
    } else blk: {
        // Create network, train it and serialize it.
        const epochs = 20;
        const batch_size = 10;
        const eta = 3.0;

        var nw = try Network.init(allocator, layers[0..], rand);
        try nw.sgd(Image, training_data, epochs, batch_size, eta, rand);

        const file = file: {
            const s = makeFilename(allocator, nw, epochs, batch_size, eta) catch "nn-xxx";
            defer allocator.free(s);
            break :file try cwd.createFile(s, .{});
        };
        defer file.close();
        try nw.serialize(file.writer());

        break :blk nw;
    };
    defer nw.deinit();

    const test_data = try getImages(allocator, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    defer allocator.free(test_data);

    try nw.validate(Image, test_data);
}

fn makeFilename(allocator: *mem.Allocator, nw: Network, epochs: usize, batch_size: usize, eta: f32) ![]const u8 {
    var buf = ArrayList(u8).init(allocator);
    const e = @floatToInt(u32, 10 * eta);
    try buf.writer().print("nn-e{}-b{}-h{}_{}-", .{ epochs, batch_size, e / 10, e % 10 });
    for (nw.layers) |l| {
        try buf.writer().print("{}-", .{l});
    }
    try buf.writer().print("T{}", .{std.time.timestamp()});
    return buf.toOwnedSlice();
}

fn getImages(allocator: *mem.Allocator, data_path: []const u8, label_path: []const u8) ![]Image {
    const cwd = fs.cwd();

    const img_file = try cwd.readFileAlloc(allocator, data_path, math.maxInt(usize));
    defer allocator.free(img_file);

    const lbl_file = try cwd.readFileAlloc(allocator, label_path, math.maxInt(usize));
    defer allocator.free(lbl_file);

    var img_fbs = io.fixedBufferStream(img_file);
    const img_reader = img_fbs.reader();

    var lbl_fbs = io.fixedBufferStream(lbl_file);
    const lbl_reader = lbl_fbs.reader();

    // Magic number
    std.debug.assert((try img_reader.readIntBig(u32)) == 0x00000803);
    std.debug.assert((try lbl_reader.readIntBig(u32)) == 0x00000801);

    // Number of images
    const number_of_images = try img_reader.readIntBig(u32);
    _ = try lbl_reader.readIntBig(u32);

    // Row / col
    std.debug.assert((try img_reader.readIntBig(u32)) == 28);
    std.debug.assert((try img_reader.readIntBig(u32)) == 28);

    var images = try ArrayList(Image).initCapacity(allocator, number_of_images);
    var i: usize = 0;
    while (i < number_of_images) : (i += 1) {
        const data = try img_reader.readBytesNoEof(28 * 28);
        const label = try lbl_reader.readByte();
        try images.append(.{
            .data = data,
            .label = label,
        });
    }

    return images.toOwnedSlice();
}

fn printImg(writer: anytype, img: *Image) !void {
    try writer.print("This is a {}:\n", .{img.label});

    var fbs = io.fixedBufferStream(img.data[0..]);
    const reader = fbs.reader();

    while (reader.readBytesNoEof(28)) |bytes| {
        var out = [_]u8{undefined} ** 200;
        var out_fbs = io.fixedBufferStream(out[0..]);
        var line_has_content = false;
        for (bytes) |c, i| {
            const s = if (c <= 100) " " else blk: {
                line_has_content = true;
                break :blk if (c <= 200) "▒" else "█";
            };
            try out_fbs.writer().print("{0s}{0s}", .{s});
        }
        if (line_has_content) {
            try writer.print("{s}\n", .{out_fbs.getWritten()});
        }
    } else |_| {}
}
