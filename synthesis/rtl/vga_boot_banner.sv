`timescale 1ns/1ps

// Generates a known-good message for the independent VGA hardware bring-up.
// Replace its output with microgpt_decode_controller.write_* after the LPU
// completes a token in hardware.
module vga_boot_banner (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       console_ready,
    output logic       write_valid,
    output logic [7:0] write_char
);
    localparam int MESSAGE_LENGTH = 33;
    logic [5:0] index;

    function automatic logic [7:0] message_at(input logic [5:0] i);
        begin
            unique case (i)
                 0: message_at = "T";  1: message_at = "I";
                 2: message_at = "N";  3: message_at = "Y";
                 4: message_at = " ";  5: message_at = "L";
                 6: message_at = "P";  7: message_at = "U";
                 8: message_at = " ";  9: message_at = "V";
                10: message_at = "G"; 11: message_at = "A";
                12: message_at = " "; 13: message_at = "R";
                14: message_at = "E"; 15: message_at = "A";
                16: message_at = "D"; 17: message_at = "Y";
                18: message_at = 8'h0a;
                19: message_at = "M"; 20: message_at = "O";
                21: message_at = "D"; 22: message_at = "E";
                23: message_at = "L"; 24: message_at = ":";
                25: message_at = " "; 26: message_at = "W";
                27: message_at = "A"; 28: message_at = "I";
                29: message_at = "T"; 30: message_at = "I";
                31: message_at = "N"; 32: message_at = "G";
                default: message_at = " ";
            endcase
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            index       <= '0;
            write_valid <= 1'b0;
            write_char  <= " ";
        end else begin
            write_valid <= 1'b0;
            if (console_ready && index < MESSAGE_LENGTH) begin
                write_valid <= 1'b1;
                write_char  <= message_at(index);
                index       <= index + 6'd1;
            end
        end
    end
endmodule
