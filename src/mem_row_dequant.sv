`timescale 1ns/1ps
`include "lpu_pkg.sv"

module mem_row_dequant (
    input  mem_row_t raw_row_i,
    output mxm_row_t fixed32_row_o
);

    function automatic logic signed [31:0] scale_fixed8_to_fixed32(
        input logic signed [7:0] lane_value,
        input logic signed [7:0] row_scale
    );
        logic signed [31:0] widened_value;
        integer shift_amount;
        begin
            widened_value = {{24{lane_value[7]}}, lane_value};
            shift_amount = row_scale;

            if (shift_amount >= 0) begin
                if (shift_amount > 23)
                    scale_fixed8_to_fixed32 = lane_value[7] ? 32'sh8000_0000 : 32'sh7fff_ffff;
                else
                    scale_fixed8_to_fixed32 = widened_value <<< shift_amount;
            end else if (-shift_amount > 30) begin
                scale_fixed8_to_fixed32 = lane_value[7] ? -32'sd1 : 32'sd0;
            end else begin
                scale_fixed8_to_fixed32 = widened_value >>> (-shift_amount);
            end
        end
    endfunction

    always_comb begin
        fixed_row_scale_t row_scale;
        fixed8_row_data_t row_data;
        row_scale = mem_row_scale(raw_row_i);
        row_data = mem_row_data(raw_row_i);
        fixed32_row_o = '0;
        for (int lane = 0; lane < MXM_SIZE; lane++) begin
            fixed32_row_o[lane*32 +: 32] = scale_fixed8_to_fixed32(
                $signed(row_data[lane*8 +: 8]),
                row_scale
            );
        end
    end

endmodule
