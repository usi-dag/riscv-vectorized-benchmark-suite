static unsigned long get_inst_count()
{
    unsigned long instr;
    instr = -1;
//    asm volatile ("rdinstret %[instr]"
//                : [instr]"=r"(instr));
    return instr;
}

static unsigned long get_cycles_count()
{
    unsigned long cycles;
    cycles = -1;
//    asm volatile ("rdcycle %[cycles]"
//                : [cycles]"=r"(cycles));
    return cycles;
}