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

const Image = @import("Image.zig");

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
    const training_data = try Image.fromFile(
        allocator,
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
    );
    defer allocator.free(training_data);

    const cwd = fs.cwd();

    // Get arg (weight file)
    const args = try std.process.argsAlloc(allocator);
    defer allocator.free(args);

    var network = if (args.len > 1) blk: {
        // Deserialize network from file.
        const filename = args[1];
        const file = cwd.openFile(filename, .{}) catch |_| {
            const stderr = io.getStdErr().writer();
            try stderr.print("Could not find or open file `{s}`.\n", .{filename});
            std.process.exit(1);
        };
        break :blk try Network.deserialize(allocator, file.reader());
    } else blk: {
        // Create network, train it and serialize it.
        const epochs = 20;
        const batch_size = 10;
        const eta = 3.0;

        var network = try Network.init(allocator, layers[0..], rand);
        try network.sgd(Image, training_data, epochs, batch_size, eta, rand);

        const file = file: {
            const path = try makeFilename(
                allocator,
                network,
                epochs,
                batch_size,
                eta,
            );
            defer allocator.free(path);
            break :file try cwd.createFile(path, .{});
        };
        defer file.close();
        try network.serialize(file.writer());

        break :blk network;
    };
    defer network.deinit();

    const test_data = try Image.fromFile(
        allocator,
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
    );
    defer allocator.free(test_data);

    try network.validate(Image, test_data);
}

fn makeFilename(
    allocator: *mem.Allocator,
    network: Network,
    epochs: usize,
    batch_size: usize,
    eta: f32,
) ![]const u8 {
    var buf = ArrayList(u8).init(allocator);
    const e = @floatToInt(u32, 10 * eta);
    try buf.writer().print("nn-e{}-b{}-h{}_{}-", .{ epochs, batch_size, e / 10, e % 10 });
    for (network.layers) |l| {
        try buf.writer().print("{}-", .{l});
    }
    try buf.writer().print("{}", .{std.time.timestamp()});
    return buf.toOwnedSlice();
}
