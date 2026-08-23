import pya


layout = pya.Layout()
layout.dbu = 0.001

top = layout.create_cell("lpulite_showcase")


def box(layer, x1, y1, x2, y2):
    top.shapes(layout.layer(layer, 20)).insert(
        pya.Box(*(round(v / layout.dbu) for v in (x1, y1, x2, y2)))
    )


def via(layer, x, y, size=8):
    half = size / 2
    top.shapes(layout.layer(layer, 44)).insert(
        pya.Box(
            *(round(v / layout.dbu) for v in (x - half, y - half, x + half, y + half))
        )
    )


# The coordinates match macro_placement.cfg and the 4088.56 x 4099.28 um die.
left_x = [100, 575, 1050]
right_x = [2575, 3050, 3525]
rows_y = [150, 950, 1750, 2550, 3350]
for x in left_x + right_x:
    for y in rows_y:
        # Accurate 455.3 x 446.46 um SRAM footprint with a presentation-level
        # bitcell-bank texture. The full routed GDS retains transistor detail.
        box(70, x, y, x + 455.3, y + 446.46)
        for stripe in range(16):
            sx = x + 18 + stripe * 26
            box(68, sx, y + 22, sx + 12, y + 424)
        box(69, x + 18, y + 28, x + 437, y + 52)
        box(69, x + 18, y + 394, x + 437, y + 418)

# Central architectural floorplan. These are presentation regions, not a
# replacement for the routed implementation used for area/timing estimates.
center_x1, center_x2 = 1540, 2545

# SXM / activation front end.
box(68, center_x1, 250, center_x2, 650)
for i in range(8):
    x1 = center_x1 + 35 + i * 118
    box(69, x1, 300, x1 + 82, 600)

# 8 x 8 MXM array.
mx_x, mx_y, tile, gap = 1640, 1420, 92, 13
for row in range(8):
    for col in range(8):
        x1 = mx_x + col * (tile + gap)
        y1 = mx_y + row * (tile + gap)
        box(68, x1, y1, x1 + tile, y1 + tile)
        via(68, x1 + tile / 2, y1 + tile / 2, 12)

# Eight VXM lanes.
for lane in range(8):
    y1 = 2475 + lane * 82
    box(70, 1600, y1, 2485, y1 + 52)

# ICU/control and host interface.
box(71, 1600, 3270, 2485, 3750)
for i in range(12):
    x = 1630 + i * 71
    box(69, x, 3320, x + 35, 3700)

# Westbound int8 and eastbound int32 presentation buses.
for i in range(8):
    y = 1240 + i * 13
    box(72, 130, y, center_x1, y + 7)
    box(72, center_x2, y, 3960, y + 7)

# Central north/south spines and visible power frame.
for i in range(8):
    x = 1580 + i * 125
    box(71, x, 680, x + 14, 3850)
for x1, y1, x2, y2 in [
    (45, 45, 4043, 75),
    (45, 4024, 4043, 4054),
    (45, 75, 75, 4024),
    (4013, 75, 4043, 4024),
]:
    box(72, x1, y1, x2, y2)

layout.write("asic/lpulite_showcase.gds")
