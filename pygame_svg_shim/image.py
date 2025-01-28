import pygame_svg_shim.surface as _surface

PADDING = 10

def save(surface: _surface.Surface, filename: str):
    if not filename.endswith(".svg"):
        raise ValueError("pygame_svg_shim.image.save can only save to the svg format")
    # calculate size
    x0 = 10000000000000000000000000
    x1 = -10000000000000000000000000
    y0 = 10000000000000000000000000
    y1 = -10000000000000000000000000
    if len(surface.draw_calls) == 0:
        x0 = x1 = y0 = y1 = 0
    else:
        for draw_call in surface.draw_calls:
            x0_, y0_, x1_, y1_ = draw_call.bounds()
            x0 = min(x0, x0_)
            x1 = max(x1, x1_)
            y0 = min(y0, y0_)
            y1 = max(y1, y1_)

        for draw_call in surface.draw_calls:
            draw_call.move((-x0+PADDING, -y0+PADDING))

    for draw_call in surface.draw_calls:
        draw_call.expand_percentages(x1-x0 + PADDING*2, y1-y0 + PADDING*2)

    calls = "\n\t".join(str(dc) for dc in surface.draw_calls)

    out: str = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg version="1.1"
	xmlns="http://www.w3.org/2000/svg"
	width="{x1-x0 + PADDING*2}"
	height="{y1-y0 + PADDING*2}"
	style="background-color:{surface.background_color}"
	><rect width="{x1-x0 + PADDING*2}" height="{y1-y0 + PADDING*2}" fill="{surface.background_color}" />{calls}
</svg>"""

    with open(filename, "w") as f:
        f.write(out)