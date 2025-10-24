// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vcounter.h for the primary calling header

#include "Vcounter.h"
#include "Vcounter__Syms.h"

//==========

VL_CTOR_IMP(Vcounter) {
    Vcounter__Syms* __restrict vlSymsp = __VlSymsp = new Vcounter__Syms(this, name());
    Vcounter* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void Vcounter::__Vconfigure(Vcounter__Syms* vlSymsp, bool first) {
    if (false && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
    if (false && this->__VlSymsp) {}  // Prevent unused
    Verilated::timeunit(-12);
    Verilated::timeprecision(-12);
}

Vcounter::~Vcounter() {
    VL_DO_CLEAR(delete __VlSymsp, __VlSymsp = NULL);
}

void Vcounter::_eval_initial(Vcounter__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcounter::_eval_initial\n"); );
    Vcounter* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
    vlTOPp->__Vclklast__TOP__rst = vlTOPp->rst;
}

void Vcounter::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcounter::final\n"); );
    // Variables
    Vcounter__Syms* __restrict vlSymsp = this->__VlSymsp;
    Vcounter* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vcounter::_eval_settle(Vcounter__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcounter::_eval_settle\n"); );
    Vcounter* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vcounter::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vcounter::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    rst = VL_RAND_RESET_I(1);
    result = VL_RAND_RESET_I(5);
    { int __Vi0=0; for (; __Vi0<1; ++__Vi0) {
            __Vm_traceActivity[__Vi0] = VL_RAND_RESET_I(1);
    }}
}
