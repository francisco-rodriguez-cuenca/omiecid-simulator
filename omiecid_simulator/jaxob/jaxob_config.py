# Modified by Francisco Rodriguez-Cuenca, 2026
import omiecid_simulator.jaxob.jaxob_constants as cst

from dataclasses import dataclass,field





@dataclass(frozen=True)
class JAXLOB_Configuration:
    maxint : int = cst.MaxInt._64_Bit_Signed.value
    init_id :int = cst.INITID
    book_depth: int = 10
    cancel_mode: int= cst.CancelMode.INCLUDE_INITS.value
    type_4_interpretation: int = cst.Type4Interpretation.IOC.value
    seed: int =cst.SEED
    nTrades : int=cst.NTRADE_CAP
    nOrders : int =cst.NORDER_CAP
    simulator_mode: int = cst.SimulatorMode.GENERAL_EXCHANGE.value
    empty_slot_val: int = cst.EMPTY_SLOT
    debug_mode: bool = False
    check_book_fill: bool = True #No major impact on performance, necessary as full book happens quite often.
