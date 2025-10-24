module counter (
    input clk,
    input rst,
    output  reg [4:0] result
);
    

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result <= 5'b0;
        end else begin
            result <= result + 1;
        end
        
    end
endmodule