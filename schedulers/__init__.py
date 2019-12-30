import logging 
from .schedulers import *
import copy

logger = logging.getLogger('ptsemseg')

key2scheduler = {'constant_lr': ConstantLR,
                 'poly_lr': PolynomialLR,
                 'multi_step': MultiStepLR,
                 'cosine_annealing': CosineAnnealingLR,
                 'exp_lr': ExponentialLR}


def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        logger.info('Using No LR Scheduling')
        return ConstantLR(optimizer)
    scheduler_dict_old = copy.deepcopy(scheduler_dict)
    s_type = scheduler_dict['name']
    scheduler_dict_old.pop('name')

    logging.info('Using {} scheduler with {} params'.format(s_type,
                                                            scheduler_dict_old))

    warmup_dict = {} 
    if 'warmup_iters' in scheduler_dict_old:
        # This can be done in a more pythonic way... 
        warmup_dict['warmup_iters'] = scheduler_dict_old.get('warmup_iters', 100)
        warmup_dict['mode'] = scheduler_dict_old.get('warmup_mode', 'linear')
        warmup_dict['gamma'] = scheduler_dict_old.get('warmup_factor', 0.2)

        logger.info('Using Warmup with {} iters {} gamma and {} mode'.format(
                                        warmup_dict['warmup_iters'],
                                        warmup_dict['gamma'],
                                        warmup_dict['mode']))

        scheduler_dict_old.pop('warmup_iters', None) 
        scheduler_dict_old.pop('warmup_mode', None)
        scheduler_dict_old.pop('warmup_factor', None) 

        base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict_old)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)
        

    return key2scheduler[s_type](optimizer, **scheduler_dict_old)
