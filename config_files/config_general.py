"""
环境变量驱动的 config_general：支持多进程/多卡并行 + 实验变体隔离。
  DUALV4_DS       数据集配置 (MSL/SWAT/WADI/ASD/SMD/SMAP)，默认 WADI
  DUALV4_DEVICE   覆盖 self.device (cuda:0 / cuda:1)
  DUALV4_EPOCHS   覆盖 self.epochs
  DUALV4_TAG      实验标签：给 save/output/log/score/results 目录加子目录，隔离各实验产物
  DUALV4_PREVLEN  覆盖 previous_steps_length (A1 参数实验)
  DUALV4_WINDOW   覆盖 window_size          (A2 参数实验)
  DUALV4_DIFFSTEP 覆盖 max_diffusion_step    (C5 消融)
不设任何环境变量时 == 原 WADI 配置。
"""
import os, importlib

_MAP = {'MSL':'config_MSL','SWAT':'config_SWAT','WADI':'config_WADI',
        'ASD':'config_ASD','SMD':'config_SMD','SMAP':'config_SMAP','PSM':'config_PSM'}
_ds = os.environ.get('DUALV4_DS', 'WADI').upper()
_BaseConfig = importlib.import_module('config_files.' + _MAP[_ds]).Config


class Config(_BaseConfig):
    def __init__(self):
        super().__init__()
        g = os.environ.get
        if g('DUALV4_DEVICE'):   self.device = g('DUALV4_DEVICE')
        if g('DUALV4_EPOCHS'):   self.epochs = int(g('DUALV4_EPOCHS'))
        if g('DUALV4_PREVLEN'):  self.previous_steps_length = int(g('DUALV4_PREVLEN'))
        if g('DUALV4_WINDOW'):   self.window_size = int(g('DUALV4_WINDOW'))
        if g('DUALV4_BATCH'):    self.batch_size = int(g('DUALV4_BATCH'))  # 大T配置降batch防OOM
        if g('DUALV4_DIFFSTEP') is not None and g('DUALV4_DIFFSTEP') != '':
            self.max_diffusion_step = int(g('DUALV4_DIFFSTEP'))
        tag = g('DUALV4_TAG')
        if tag:
            for a in ('save_dir', 'output_dir', 'log_dir', 'score_dir', 'results_dir'):
                setattr(self, a, os.path.join(getattr(self, a), tag))
