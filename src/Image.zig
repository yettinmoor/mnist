const std = @import("std");
const mem = std.mem;

const Image = @This();

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

pub fn fromFile(allocator: *mem.Allocator, data_path: []const u8, label_path: []const u8) ![]Image {
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
    {
        const img_magic = try img_reader.readIntBig(u32);
        const lbl_magic = try img_reader.readIntBig(u32);
        std.debug.assert(img_magic == 0x00000803 and lbl_magic == 0x00000801);
    }

    // Number of images
    const number_of_images = try img_reader.readIntBig(u32);
    _ = try lbl_reader.readIntBig(u32);

    // Row / col
    {
        const rows = try img_reader.readIntBig(u32);
        const cols = try img_reader.readIntBig(u32);
        std.debug.assert(rows == 28 and cols == 28);
    }

    var images = try ArrayList(Image).initCapacity(allocator, number_of_images);
    images.items.len = number_of_images;
    for (images.items) |*img| {
        const data = try img_reader.readBytesNoEof(28 * 28);
        const label = try lbl_reader.readByte();
        img.* = .{
            .data = data,
            .label = label,
        };
    }

    return images.toOwnedSlice();
}

pub fn printImg(writer: anytype, img: *Image) !void {
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
