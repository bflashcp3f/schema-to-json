
from schema2json.tasks.mltables import MLTables
from schema2json.tasks.chemtables import ChemTables
from schema2json.tasks.discomat import DiSCoMaT
from schema2json.tasks.swde import SWDE

def get_task(args):
    if args.task.startswith('mltables'):
        return MLTables(args)
    elif args.task.startswith('chemtables'):
        return ChemTables(args)
    elif args.task == 'discomat':
        return DiSCoMaT(args)
    elif args.task == 'swde':
        return SWDE(args)
    else:
        raise NotImplementedError(f'Task "{args.task}" not implemented')