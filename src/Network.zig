const std = @import("std");
const io = std.io;
const fs = std.fs;
const mem = std.mem;
const math = std.math;
const heap = std.heap;
const ArrayList = std.ArrayList;

const matrix = @import("matrix.zig");
const Matf = matrix.Matrix(f32);

const Network = @This();

const MAGIC = 0xfafa1331;

allocator: *mem.Allocator,

layers: []const u32,
weights: []Matf,
biases: []Matf,

pub fn deserialize(allocator: *mem.Allocator, reader: anytype) !Network {
    const magic = try reader.readIntLittle(u32);
    if (magic != MAGIC) {
        return error.InvalidFile;
    }

    const size = try reader.readIntLittle(u32);
    var layers = try allocator.alloc(u32, size);
    var weights = try allocator.alloc(Matf, size - 1);
    var biases = try allocator.alloc(Matf, size - 1);

    var i: u32 = 0;
    while (i < size) : (i += 1) {
        layers[i] = try reader.readIntLittle(u32);
    }

    i = 0;
    while (i < size - 1) : (i += 1) {
        const m = try reader.readIntLittle(u32);
        const n = try reader.readIntLittle(u32);
        var w = try Matf.init(allocator, m, n);
        for (w.data) |*x| {
            const int = try reader.readIntLittle(u32);
            x.* = @bitCast(f32, int);
        }
        weights[i] = w;
    }

    i = 0;
    while (i < size - 1) : (i += 1) {
        const m = try reader.readIntLittle(u32);
        const n = try reader.readIntLittle(u32);
        var b = try Matf.init(allocator, m, n);
        for (b.data) |*x| {
            const int = try reader.readIntLittle(u32);
            x.* = @bitCast(f32, int);
        }
        biases[i] = b;
    }

    return Network{
        .allocator = allocator,
        .layers = layers,
        .weights = weights,
        .biases = biases,
    };
}

pub fn init(
    allocator: *mem.Allocator,
    layers: []const u32,
    rand: *std.rand.Random,
) !Network {
    var weights = try allocator.alloc(Matf, layers.len - 1);
    var biases = try allocator.alloc(Matf, layers.len - 1);

    for (layers[0 .. layers.len - 1]) |l0, i| {
        const l1 = layers[i + 1];
        weights[i] = try Matf.init(allocator, l1, l0);
        biases[i] = try Matf.init(allocator, l1, 1);
        for (weights[i].data) |*w| {
            w.* = rand.floatNorm(f32);
        }
        for (biases[i].data) |*b| {
            b.* = rand.floatNorm(f32);
        }
    }

    const layers_copy = try allocator.dupe(u32, layers);

    return Network{
        .allocator = allocator,
        .layers = layers_copy,
        .weights = weights,
        .biases = biases,
    };
}

pub fn deinit(nw: *Network) void {
    for (nw.weights) |w| {
        w.deinit();
    }
    for (nw.biases) |b| {
        b.deinit();
    }
    nw.allocator.free(nw.layers);
    nw.allocator.free(nw.weights);
    nw.allocator.free(nw.biases);
    nw.* = undefined;
}

pub fn feed(nw: *Network, comptime T: type, data: T) !Matf {
    var arena = std.heap.ArenaAllocator.init(nw.allocator);
    defer arena.deinit();
    const allocator = &arena.allocator;

    var input = try data.toInput(allocator);
    var output: Matf = undefined;
    for (nw.weights) |w, i| {
        // a = sigma(w * a' + b)
        var zs = try nw.biases[i].clone(allocator);
        const w_mul_a = try w.mul(input, allocator);
        zs.add(w_mul_a);
        output = try zs.apply(allocator, f32, sigmoid);
        input = output;
    }
    return output.clone(nw.allocator);
}

pub fn trainBatch(nw: *Network, comptime T: type, batch: []T, eta: f32) !void {
    var arena = std.heap.ArenaAllocator.init(nw.allocator);
    defer arena.deinit();
    const allocator = &arena.allocator;

    var nabla_w = try allocator.alloc(Matf, nw.layers.len - 1);
    var nabla_b = try allocator.alloc(Matf, nw.layers.len - 1);

    for (nw.layers[0 .. nw.layers.len - 1]) |l0, i| {
        const l1 = nw.layers[i + 1];
        nabla_w[i] = try Matf.init(allocator, l1, l0);
        nabla_b[i] = try Matf.init(allocator, l1, 1);
        mem.set(f32, nabla_w[i].data, 0);
        mem.set(f32, nabla_b[i].data, 0);
    }

    mem.reverse(Matf, nabla_w);
    mem.reverse(Matf, nabla_b);

    for (batch) |data| {
        var zs = try allocator.alloc(Matf, nw.layers.len - 1);
        var activations = try allocator.alloc(Matf, nw.layers.len);

        // For each input, calculate all z values and sigmoid activations.
        {
            var a = try data.toInput(allocator);
            activations[0] = a;
            for (nw.weights) |w, i| {
                // a = sigma(w * a' + b)
                var z = try nw.biases[i].clone(allocator);
                const w_mul_a = try w.mul(a, allocator);
                z.add(w_mul_a);
                zs[i] = z;

                a = try z.apply(allocator, f32, sigmoid);
                activations[i + 1] = a;
            }
        }

        mem.reverse(Matf, zs);
        mem.reverse(Matf, activations);

        // Output layer error:
        //   δ = ∇ C ⊙ σ'(z)
        var delta = try activations[0].clone(allocator);
        delta.data[data.label] -= 1.0;

        // Previous layer error:
        //   δ' = (w * δ) ⊙ σ'(z)
        for (zs) |z, i| {
            if (i > 0) {
                var ws_t = try nw.weights[nw.weights.len - i].transpose(allocator);
                delta = try ws_t.mul(delta, allocator);
            }

            var sig_p = try z.apply(allocator, f32, sigmoid_prime);
            delta.mulElem(sig_p);

            // ∇_b(C) = δ
            // ∇_W(C) = a' * δ
            nabla_b[i].add(delta);
            nabla_w[i].add(blk: {
                var a_t = try activations[i + 1].transpose(allocator);
                break :blk try delta.mul(a_t, allocator);
            });
        }
    }

    mem.reverse(Matf, nabla_w);
    mem.reverse(Matf, nabla_b);

    const c = eta / @intToFloat(f32, batch.len);
    for (nw.weights) |*w, i| {
        nabla_w[i].mulScalar(c);
        w.sub(nabla_w[i]);
    }
    for (nw.biases) |*b, i| {
        nabla_b[i].mulScalar(c);
        b.sub(nabla_b[i]);
    }
}

pub fn sgd(
    nw: *Network,
    comptime T: type,
    data: []T,
    epochs: usize,
    batch_size: usize,
    eta: f32,
    rand: *std.rand.Random,
) !void {
    const stdout = io.getStdOut().writer();

    const training_size = 5 * data.len / 6;
    var epoch: usize = 0;
    while (epoch < epochs) : (epoch += 1) {
        try stdout.print("Processing epoch {}...", .{epoch + 1});

        rand.shuffle(T, data);

        var processed: usize = 0;
        while (processed < training_size) : (processed += batch_size) {
            try stdout.print("\rProcessing batch {}/{}...", .{ processed / batch_size, training_size / batch_size });
            const batch = data[processed .. processed + batch_size];
            try nw.trainBatch(T, batch, eta);
        }

        var correct: usize = 0;
        const eval_data = data[training_size..];
        for (eval_data) |ed| {
            const output = try nw.feed(T, ed);
            defer output.deinit();
            const guess = guessIndex(output);
            correct += @boolToInt(guess == ed.label);
        }

        var buf = [_]u8{undefined} ** 256;
        var fbs = io.fixedBufferStream(buf[0..]);
        const percent_correct = 100 * correct / eval_data.len;

        try fbs.writer().print("\rEpoch {}: {}/{} ", .{ epoch + 1, correct, eval_data.len });
        const pad = fbs.getPos() catch unreachable;
        try stdout.writeAll(fbs.getWritten());
        try stdout.writeByteNTimes(' ', 25 - pad);

        var i: usize = 0;
        while (i < percent_correct) : (i += 4) {
            try stdout.writeAll("█");
        }
        while (i < 100) : (i += 4) {
            try stdout.writeAll("░");
        }
        try stdout.print(" [{}%]\n", .{percent_correct});
    }
}

pub fn validate(nw: *Network, comptime T: type, test_data: []const T) !void {
    const stdout = io.getStdOut().writer();
    try stdout.writeAll("Running network on test data...\n");

    var correct: usize = 0;
    for (test_data) |td| {
        const output = try nw.feed(T, td);
        defer output.deinit();
        const guess = guessIndex(output);
        correct += @boolToInt(guess == td.label);
    }

    try stdout.print("Result: {} / {}\n", .{ correct, test_data.len });
}

pub fn serialize(nw: Network, writer: anytype) !void {
    try writer.writeIntLittle(u32, MAGIC);
    try writer.writeIntLittle(u32, @intCast(u32, nw.layers.len));
    for (nw.layers) |l| {
        try writer.writeIntLittle(u32, l);
    }
    for (nw.weights) |w| {
        try writer.writeIntLittle(u32, w.m);
        try writer.writeIntLittle(u32, w.n);
        for (w.data) |x| {
            try writer.writeIntLittle(u32, @bitCast(u32, x));
        }
    }
    for (nw.biases) |b| {
        try writer.writeIntLittle(u32, b.m);
        try writer.writeIntLittle(u32, b.n);
        for (b.data) |x| {
            try writer.writeIntLittle(u32, @bitCast(u32, x));
        }
    }
}

fn guessIndex(output: Matf) usize {
    var guess: f32 = 0;
    var guess_index: usize = 0;
    for (output.data) |x, i| {
        if (x > guess) {
            guess = x;
            guess_index = i;
        }
    }
    return guess_index;
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + math.exp(-x));
}

fn sigmoid_prime(x: f32) f32 {
    const s = sigmoid(x);
    return s * (1 - s);
}
