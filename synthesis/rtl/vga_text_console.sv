`timescale 1ns/1ps

// 80 x 30 text-only VGA renderer for the DE1-SoC.
//
// A divide-by-two register derives a 25 MHz pixel clock from the board's 50 MHz
// oscillator. This is close to the nominal 25.175 MHz 640x480@60 Hz rate and
// is accepted by most VGA displays. A PLL can replace pixel_clk later if a
// particular monitor requires the exact VESA clock.
module vga_text_console (
    input  logic       clk_50,
    input  logic       rst_n,
    input  logic       write_valid,
    input  logic [7:0] write_char,
    output logic       ready,
    output logic [7:0] vga_r,
    output logic [7:0] vga_g,
    output logic [7:0] vga_b,
    output logic       vga_clk,
    output logic       vga_blank_n,
    output logic       vga_hs,
    output logic       vga_vs
);
    localparam int COLS       = 80;
    localparam int ROWS       = 30;
    localparam int CELL_COUNT = COLS * ROWS;

    // A separate registered display read port below makes this an M10K true
    // dual-port RAM: one port updates terminal text, the other scans video.
    (* ramstyle = "M10K, no_rw_check" *) logic [7:0] text_ram [0:CELL_COUNT-1];
    logic [11:0] clear_index;
    logic        clearing;
    logic        pixel_clk;
    logic [9:0]  h_count;
    logic [9:0]  v_count;
    logic [6:0]  cursor_col;
    logic [5:0]  cursor_row;

    logic [11:0] char_address;
    logic [7:0]  current_char;
    logic [34:0] glyph_word;
    logic [4:0]  glyph_row_bits;
    logic        glyph_pixel;
    logic        active_video;

    function automatic logic [34:0] glyph_pattern(input logic [7:0] character);
        begin
            // 5x7 glyphs. Lowercase is deliberately rendered as uppercase so
            // a MicroGPT character vocabulary needs no font conversion.
            unique case (character)
                "A", "a": glyph_pattern = 35'b01110_10001_10001_11111_10001_10001_10001;
                "B", "b": glyph_pattern = 35'b11110_10001_10001_11110_10001_10001_11110;
                "C", "c": glyph_pattern = 35'b01110_10001_10000_10000_10000_10001_01110;
                "D", "d": glyph_pattern = 35'b11110_10001_10001_10001_10001_10001_11110;
                "E", "e": glyph_pattern = 35'b11111_10000_10000_11110_10000_10000_11111;
                "F", "f": glyph_pattern = 35'b11111_10000_10000_11110_10000_10000_10000;
                "G", "g": glyph_pattern = 35'b01110_10001_10000_10111_10001_10001_01110;
                "H", "h": glyph_pattern = 35'b10001_10001_10001_11111_10001_10001_10001;
                "I", "i": glyph_pattern = 35'b01110_00100_00100_00100_00100_00100_01110;
                "J", "j": glyph_pattern = 35'b00111_00010_00010_00010_00010_10010_01100;
                "K", "k": glyph_pattern = 35'b10001_10010_10100_11000_10100_10010_10001;
                "L", "l": glyph_pattern = 35'b10000_10000_10000_10000_10000_10000_11111;
                "M", "m": glyph_pattern = 35'b10001_11011_10101_10101_10001_10001_10001;
                "N", "n": glyph_pattern = 35'b10001_11001_10101_10011_10001_10001_10001;
                "O", "o": glyph_pattern = 35'b01110_10001_10001_10001_10001_10001_01110;
                "P", "p": glyph_pattern = 35'b11110_10001_10001_11110_10000_10000_10000;
                "Q", "q": glyph_pattern = 35'b01110_10001_10001_10001_10101_10010_01101;
                "R", "r": glyph_pattern = 35'b11110_10001_10001_11110_10100_10010_10001;
                "S", "s": glyph_pattern = 35'b01111_10000_10000_01110_00001_00001_11110;
                "T", "t": glyph_pattern = 35'b11111_00100_00100_00100_00100_00100_00100;
                "U", "u": glyph_pattern = 35'b10001_10001_10001_10001_10001_10001_01110;
                "V", "v": glyph_pattern = 35'b10001_10001_10001_10001_10001_01010_00100;
                "W", "w": glyph_pattern = 35'b10001_10001_10001_10101_10101_10101_01010;
                "X", "x": glyph_pattern = 35'b10001_10001_01010_00100_01010_10001_10001;
                "Y", "y": glyph_pattern = 35'b10001_10001_01010_00100_00100_00100_00100;
                "Z", "z": glyph_pattern = 35'b11111_00001_00010_00100_01000_10000_11111;
                "0": glyph_pattern = 35'b01110_10001_10011_10101_11001_10001_01110;
                "1": glyph_pattern = 35'b00100_01100_00100_00100_00100_00100_01110;
                "2": glyph_pattern = 35'b01110_10001_00001_00010_00100_01000_11111;
                "3": glyph_pattern = 35'b11110_00001_00001_01110_00001_00001_11110;
                "4": glyph_pattern = 35'b00010_00110_01010_10010_11111_00010_00010;
                "5": glyph_pattern = 35'b11111_10000_10000_11110_00001_00001_11110;
                "6": glyph_pattern = 35'b01110_10000_10000_11110_10001_10001_01110;
                "7": glyph_pattern = 35'b11111_00001_00010_00100_01000_01000_01000;
                "8": glyph_pattern = 35'b01110_10001_10001_01110_10001_10001_01110;
                "9": glyph_pattern = 35'b01110_10001_10001_01111_00001_00001_01110;
                ".": glyph_pattern = 35'b00000_00000_00000_00000_00000_00110_00110;
                ":": glyph_pattern = 35'b00000_00110_00110_00000_00110_00110_00000;
                "-": glyph_pattern = 35'b00000_00000_00000_11111_00000_00000_00000;
                "?": glyph_pattern = 35'b01110_10001_00001_00010_00100_00000_00100;
                default: glyph_pattern = 35'b00000_00000_00000_00000_00000_00000_00000;
            endcase
        end
    endfunction

    assign ready   = !clearing;
    assign vga_clk = pixel_clk;

    // Pixel-clock divider.  The renderer runs at 25 MHz; the terminal writer
    // runs at 50 MHz through the other M10K port, so a banner or LPU writer
    // never loses every other character.
    always_ff @(posedge clk_50 or negedge rst_n) begin
        if (!rst_n) begin
            pixel_clk <= 1'b0;
        end else begin
            pixel_clk <= ~pixel_clk;
        end
    end

    // M10K port A: writer clock domain.  This intentionally has a different
    // clock from the scan port below; Quartus infers a true dual-port RAM.
    always_ff @(posedge clk_50 or negedge rst_n) begin
        if (!rst_n) begin
            clearing    <= 1'b1;
            clear_index <= '0;
            cursor_col  <= '0;
            cursor_row  <= '0;
        end else begin
            if (clearing) begin
                text_ram[clear_index] <= " ";
                if (clear_index == CELL_COUNT - 1) begin
                    clearing   <= 1'b0;
                    cursor_col <= '0;
                    cursor_row <= '0;
                end else begin
                    clear_index <= clear_index + 12'd1;
                end
            end else if (write_valid) begin
                if (write_char == 8'h0a) begin
                    cursor_col <= '0;
                    if (cursor_row == ROWS - 1) cursor_row <= '0;
                    else                        cursor_row <= cursor_row + 6'd1;
                end else begin
                    text_ram[cursor_row * COLS + cursor_col] <= write_char;
                    if (cursor_col == COLS - 1) begin
                        cursor_col <= '0;
                        if (cursor_row == ROWS - 1) cursor_row <= '0;
                        else                        cursor_row <= cursor_row + 6'd1;
                    end else begin
                        cursor_col <= cursor_col + 7'd1;
                    end
                end
            end

        end
    end

    // Scan timing is the 25 MHz VGA pixel clock.
    always_ff @(posedge pixel_clk or negedge rst_n) begin
        if (!rst_n) begin
            h_count <= '0;
            v_count <= '0;
        end else if (h_count == 10'd799) begin
            h_count <= '0;
            if (v_count == 10'd524) v_count <= '0;
            else                     v_count <= v_count + 10'd1;
        end else begin
            h_count <= h_count + 10'd1;
        end
    end

    // Dedicated synchronous scan/read port.  Keeping it separate from the
    // 50 MHz writer is the Quartus inference pattern for a dual-port M10K.
    always_ff @(posedge pixel_clk) begin
        current_char <= text_ram[char_address];
    end

    always_comb begin
        active_video = (h_count < 10'd640) && (v_count < 10'd480);
        char_address = (v_count[8:4] * COLS) + h_count[9:3];
        glyph_word = glyph_pattern(current_char);
        glyph_row_bits = glyph_word[34 - (v_count[3:1] * 5) -: 5];
        glyph_pixel = (h_count[2:0] >= 3'd1) && (h_count[2:0] <= 3'd5) &&
                      glyph_row_bits[5 - h_count[2:0]];

        vga_blank_n = active_video;
        vga_hs = !((h_count >= 10'd656) && (h_count < 10'd752));
        vga_vs = !((v_count >= 10'd490) && (v_count < 10'd492));
        vga_r = 8'h00;
        vga_g = (active_video && glyph_pixel) ? 8'hff : 8'h00;
        vga_b = (active_video && glyph_pixel) ? 8'h40 : 8'h00;
    end
endmodule
