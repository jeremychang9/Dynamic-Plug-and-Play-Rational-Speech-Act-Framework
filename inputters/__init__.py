
from inputters.strat import Inputter as strat
from inputters.strat_pp import Inputter as strat_pp
from inputters.vanilla import Inputter as vanilla
from inputters.strat_llama import Inputter as strat_llama

inputters = {
    'vanilla': vanilla,
    'strat': strat,
    'strat_pp': strat_pp, # for plug-and-play
    'strat_llama': strat_llama,
}



