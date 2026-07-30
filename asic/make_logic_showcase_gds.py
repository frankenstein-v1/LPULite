import pya


layout = pya.Layout()
layout.dbu = 0.001
top = layout.create_cell("tinylpu_logic")


def layer(number, datatype=20):
    return layout.layer(number, datatype)


def box(number, x1, y1, x2, y2, datatype=20):
    top.shapes(layer(number, datatype)).insert(
        pya.Box(*(round(v / layout.dbu) for v in (x1, y1, x2, y2)))
    )


def via(number, x, y, size=8):
    h = size / 2
    box(number, x - h, y - h, x + h, y + h, 44)


def frame(number, x1, y1, x2, y2, width=12):
    box(number, x1, y1, x2, y1 + width)
    box(number, x1, y2 - width, x2, y2)
    box(number, x1, y1, x1 + width, y2)
    box(number, x2 - width, y1, x2, y2)


def register_bank(x1, y1, columns, rows, pitch_x, pitch_y, w=15, h=22):
    for row in range(rows):
        for col in range(columns):
            box(
                71,
                x1 + col * pitch_x,
                y1 + row * pitch_y,
                x1 + col * pitch_x + w,
                y1 + row * pitch_y + h,
            )


def mac_tile(x, y, size=206):
    # MAC boundary and multiplier partial-product fabric.
    frame(68, x, y, x + size, y + size, 8)
    for bit in range(8):
        yy = y + 24 + bit * 13
        box(68, x + 20, yy, x + 112 + bit * 7, yy + 6)
        via(68, x + 28 + bit * 9, yy + 3, 7)

    # Reduction/adder tree, shown as progressively shorter metal stages.
    stage_y = y + 132
    spans = [(18, 184), (32, 172), (50, 154), (72, 136)]
    for stage, (x1, x2) in enumerate(spans):
        yy = stage_y + stage * 13
        box(69, x + x1, yy, x + x2, yy + 7)
        for node in range(2**max(0, 3 - stage)):
            nx = x + x1 + (node + 0.5) * (x2 - x1) / (2**max(0, 3 - stage))
            via(69, nx, yy + 3.5, 7)

    # Accumulator/register slice.
    for reg in range(8):
        rx = x + 19 + reg * 21
        box(70, rx, y + 184, rx + 14, y + 198)


def memory_array(x, y, width, height, banks, words=20):
    frame(70, x, y, x + width, y + height, 14)
    bank_w = (width - 36) / banks
    for bank in range(banks):
        bx = x + 18 + bank * bank_w
        box(70, bx, y + 18, bx + bank_w - 8, y + height - 18)
        for word in range(words):
            yy = y + 24 + word * (height - 48) / words
            box(68, bx + 7, yy, bx + bank_w - 15, yy + 3)
    # Decoder spine and sense-amplifier rail.
    box(69, x + 24, y + 22, x + 42, y + height - 22)
    box(69, x + 18, y + height - 44, x + width - 18, y + height - 22)


# --- 8 x 8 matrix engine ----------------------------------------------------
mx_x, mx_y = 1160, 690
tile, gap = 206, 24
frame(72, mx_x - 45, mx_y - 45, mx_x + 8 * (tile + gap) - gap + 45, mx_y + 8 * (tile + gap) - gap + 45, 18)
for row in range(8):
    for col in range(8):
        mac_tile(mx_x + col * (tile + gap), mx_y + row * (tile + gap), tile)

# Row/column data distribution networks across the MXM.
for lane in range(8):
    y = mx_y + lane * (tile + gap) + tile / 2
    box(72, 760, y - 5, mx_x + 8 * (tile + gap) + 310, y + 5)
    x = mx_x + lane * (tile + gap) + tile / 2
    box(71, x - 5, 500, x + 5, 2710)

# --- Westbound input and eastbound output lane logic ------------------------
for lane in range(8):
    y = mx_y + lane * (tile + gap) + 55
    # Eight int8 input register slices.
    register_bank(800, y, 8, 1, 24, 0, 15, 32)
    box(69, 760, y + 39, 1115, y + 48)

    # Eight int32 accumulator/output slices.
    register_bank(3075, y, 8, 4, 24, 22, 15, 15)
    box(69, 2995, y + 39, 3260, y + 48)

# --- VXM: eight visible vector lanes, each with four named-stage geometries --
vxm_y = 2730
stage_widths = [52, 62, 72, 82]
for lane_index in range(8):
    lane_x = 930 + lane_index * 300
    frame(71, lane_x, vxm_y, lane_x + 260, vxm_y + 430, 10)
    for stage in range(4):
        sy = vxm_y + 32 + stage * 92
        box(70, lane_x + 24, sy, lane_x + 236, sy + 62)
        # Internal LUT/arithmetic texture: quantize, RoPE, RMSNorm, softmax.
        for unit in range(stage_widths[stage] // 10):
            ux = lane_x + 34 + unit * 18
            box(68 + stage % 2, ux, sy + 12, ux + 10, sy + 50)
            if unit % 2 == 0:
                via(69, ux + 5, sy + 31, 6)

# MXM-to-VXM result fanout.
for lane_index in range(8):
    x = 1030 + lane_index * 300
    box(72, x - 6, 2490, x + 6, vxm_y)

# --- Real architectural memories --------------------------------------------
memory_array(80, 620, 590, 930, 9, 24)      # MEM0 bank group
memory_array(80, 1640, 590, 930, 9, 24)     # MEM1 bank group
memory_array(3440, 620, 650, 1950, 12, 32)  # 96-bit instruction memory

# ICU / sequencer state and decode forest.
frame(72, 80, 2680, 650, 3160, 16)
register_bank(115, 2730, 12, 8, 38, 42, 22, 25)
for decoder in range(7):
    y = 3080 - decoder * 42
    box(69, 430 + decoder * 18, y, 620 - decoder * 18, y + 9)

# Shared control distribution.
for index in range(6):
    y = 320 + index * 34
    box(71, 120, y, 4010, y + 10)
    for x in range(450, 3900, 350):
        via(71, x, y + 5, 9)

# Perimeter data ring and repeated bus taps.
frame(72, 35, 275, 4135, 3215, 24)
for tap in range(16):
    x = 180 + tap * 245
    box(72, x, 285, x + 9, 430)
    via(71, x + 4.5, 330, 10)

layout.write("asic/tinylpu_logic_showcase.gds")
