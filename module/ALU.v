module ALU (
    input clk,
    input rst,
    input [31:0] operate_select,
    input [31:0] operate_a,
    input [31:0] operate_b,
    output [31:0] result,
    output  zero_flag,
);
    always  (@ posedge clk or posedge rst) begin
        if (rst) begin
            result <= 32'b0;
            zero_flag <= 1'b0;
        end else begin
            case (operate_select)
                32'b0000: result <= operate_a + operate_b; // 加法
                32'b0001: result <= operate_a - operate_b; // 减法
                32'b0010: result <= operate_a & operate_b; // 与
                32'b0011: result <= operate_a | operate_b; // 或
                32'b0100: result <= operate_a ^ operate_b; // 异或
                32'b0101: result <= ~operate_a;            // 非
                32'b0110: result <= operate_a << 1;       // 左移
                32'b0111: result <= operate_a >> 1;       // 右移
                default: result <= 32'b0;                  // 默认值
            endcase
            if (result == 32'b0) begin
                zero_flag <= 1'b1;
            end else begin
                zero_flag <= 1'b0;
            end
        end
    end
endmodule