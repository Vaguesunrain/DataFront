// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vcounter__Syms.h"


void Vcounter::traceChgTop0(void* userp, VerilatedVcd* tracep) {
    Vcounter__Syms* __restrict vlSymsp = static_cast<Vcounter__Syms*>(userp);
    Vcounter* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    {
        vlTOPp->traceChgSub0(userp, tracep);
    }
}

void Vcounter::traceChgSub0(void* userp, VerilatedVcd* tracep) {
    Vcounter__Syms* __restrict vlSymsp = static_cast<Vcounter__Syms*>(userp);
    Vcounter* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    vluint32_t* const oldp = tracep->oldp(vlSymsp->__Vm_baseCode + 1);
    if (false && oldp) {}  // Prevent unused
    // Body
    {
        tracep->chgBit(oldp+0,(vlTOPp->clk));
        tracep->chgBit(oldp+1,(vlTOPp->rst));
        tracep->chgCData(oldp+2,(vlTOPp->result),5);
    }
}

void Vcounter::traceCleanup(void* userp, VerilatedVcd* /*unused*/) {
    Vcounter__Syms* __restrict vlSymsp = static_cast<Vcounter__Syms*>(userp);
    Vcounter* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlSymsp->__Vm_activity = false;
        vlTOPp->__Vm_traceActivity[0U] = 0U;
    }
}
