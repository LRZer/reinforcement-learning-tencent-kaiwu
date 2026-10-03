"""Regenerate bilingual repository figures from the archived facts and code."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from agent_ppo.conf.conf import Config

OUT = ROOT/'docs/figures'
OUT.mkdir(parents=True, exist_ok=True)
RESULT = json.loads((ROOT/'data/results/competition.json').read_text(encoding='utf-8'))
SCHEMA = json.loads((ROOT/'data/metadata/feature_schema.json').read_text(encoding='utf-8'))
CHECKPOINT = json.loads((ROOT/'data/metadata/checkpoint_inspection.json').read_text(encoding='utf-8'))
INK = '#142b43'; TEAL = '#087f8c'; BLUE = '#3563b9'; GOLD = '#c28a25'; MUTED = '#586d83'
PALE = '#f1f6fa'; LINE = '#9eb1c2'; GREEN = '#16805b'; RED = '#bf4b51'

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for font in ['Microsoft YaHei','Noto Sans CJK SC','SimHei']:
    if font in available_fonts:
        plt.rcParams['font.family'] = [font,'DejaVu Sans']
        break
else:
    raise RuntimeError('A Chinese font is required: Microsoft YaHei, Noto Sans CJK SC or SimHei.')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'path'


def choose(zh,en,lang):
    return zh if lang == 'zh' else en


def canvas(title, subtitle, height=9):
    fig = plt.figure(figsize=(16,height), facecolor='white')
    ax = fig.add_axes([.04,.06,.92,.88]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
    ax.text(0,99,title,color=INK,fontsize=23,fontweight='bold',va='top')
    ax.text(0,91,subtitle,color=MUTED,fontsize=11,va='top')
    return fig,ax


def box(ax,x,y,w,h,label,color=TEAL,fs=12,fill=PALE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.35,rounding_size=1.1',linewidth=1.3,edgecolor=color,facecolor=fill))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',color=INK,fontsize=fs,linespacing=1.6)


def arrow(ax,start,end,color=LINE,style='-|>',connectionstyle='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=15,color=color,linewidth=1.4,connectionstyle=connectionstyle))


def save(fig,name,lang):
    for ext in ['svg','png']:
        path = OUT/f'{name}.{lang}.{ext}'
        fig.savefig(path,dpi=160,facecolor='white')
        if ext == 'svg':
            content = '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n'
            path.write_text(content, encoding='utf-8', newline='\n')
    plt.close(fig)


def results(lang):
    fig,ax = canvas(choose('比赛结果与证据','Competition results and evidence',lang),choose('第十七届中国大学生服务外包创新创业大赛 | 课程报告第 3-4 页','17th student service outsourcing competition | Course report, pp. 3-4',lang),8)
    metrics = [
        ('37 / 158',choose('初赛排名 / 队伍总数','Preliminary rank / teams',lang),TEAL),
        ('997.15',choose('打榜平均分','Leaderboard average score',lang),BLUE),
        ('150 / 150',choose('完成局数 / 总局数','Completed / total episodes',lang),TEAL),
        (choose('区域三等奖','Regional Third Prize',lang),choose('人工智能专项赛道','Artificial Intelligence Track',lang),GOLD),
    ]
    for i,(value,label,color) in enumerate(metrics):
        x = 1+i*25
        box(ax,x,58,22,25,'',color)
        ax.text(x+11,73,value,ha='center',va='center',fontsize=18 if i==3 else 26,fontweight='bold',color=color)
        ax.text(x+11,63,label,ha='center',va='center',fontsize=10.5,color=MUTED)
    ax.text(0,46,choose('排名位置（1 为最高排名）','Rank position (1 is the highest rank)',lang),fontsize=13,color=INK)
    ax.plot([2,97],[33,33],color='#dce6ee',linewidth=15,solid_capstyle='round')
    x = 2+(RESULT['preliminary_rank']-1)/(RESULT['preliminary_teams']-1)*95
    ax.scatter([x],[33],s=330,color=TEAL,zorder=3,edgecolors='white',linewidths=2)
    ax.text(x,38,'37',ha='center',color=TEAL,fontweight='bold',fontsize=16)
    ax.text(2,26,'1',ha='center',color=MUTED); ax.text(97,26,'158',ha='center',color=MUTED)
    ax.text(0,14,choose('初赛未进入决赛；区域奖项为独立记录。','The team did not advance to the final; the regional award is recorded separately.',lang),fontsize=12,color=INK)
    ax.text(0,5,choose('排名总人数来自报告文字；分数和完成局数来自打榜截图；奖项来自证书。','Team count: report text. Score and episode count: leaderboard screenshot. Award: certificate.',lang),fontsize=10.5,color=MUTED)
    save(fig,'results',lang)


def architecture(lang):
    fig,ax = canvas(choose('Actor-Critic 网络结构','Actor-Critic network architecture',lang),choose('3601 维输入 · 75,814 个参数 · 检查点 34055 与网络严格匹配','3601 input dimensions · 75,814 parameters · Checkpoint 34055 matches the network strictly',lang),10)
    rows = [('英雄','Hero',14),('怪物','Monsters',16),('宝箱','Treasures',14),('加速增益','Buffs',8),('地形拓扑','Topology',13),('历史记忆','Memory',8)]
    ax.text(1,85,choose('结构化输入（73 维）','Structured inputs (73D)',lang),fontsize=12,color=MUTED)
    ax.text(30,85,choose('独立特征编码','Separate encoders',lang),fontsize=12,color=MUTED)
    for i,(zh,en,dim) in enumerate(rows):
        y = 73-i*10
        box(ax,1,y,20,7,f'{choose(zh,en,lang)} · {dim}D',TEAL,11)
        box(ax,29,y,19,7,f'MLP {dim} → 32 → 32',BLUE,10)
        arrow(ax,(21,y+3.5),(28.5,y+3.5))
    box(ax,1,3,47,14,choose('局部地图 8 × 21 × 21\nCNN 8 → 16 → 32 → 64 → 64D','Local map 8 × 21 × 21\nCNN 8 → 16 → 32 → 64 → 64D',lang),BLUE,12)
    box(ax,57,69,40,12,choose('英雄 32D + 地图 64D → 门控网络\n96 → 32 → 5，Softmax 权重','Hero 32D + map 64D → type gate\n96 → 32 → 5, Softmax weights',lang),TEAL,12)
    arrow(ax,(48.5,76.5),(56.5,76.5))
    ax.text(59,84,choose('地图嵌入也作为门控输入','Map embedding also feeds the gate',lang),fontsize=10,color=MUTED)
    box(ax,57,43,40,13,choose('5 个上下文分支加权求和\n怪物 / 宝箱 / BUFF / 拓扑 / 记忆 → 32D','Weighted sum of 5 context branches\nMonsters / treasures / buffs / topology / memory → 32D',lang),TEAL,11)
    arrow(ax,(77,68.5),(77,56.5),TEAL)
    ax.plot([52,52],[26.5,66.5],color=BLUE,linewidth=1.4)
    for i in range(1,6):
        y = 76.5-i*10
        ax.plot([48.5,52],[y,y],color=BLUE,linewidth=1.4)
    arrow(ax,(52,49.5),(56.5,49.5),BLUE)
    ax.text(63,58,choose('上下文各分支均为 32D','Each context branch is 32D',lang),fontsize=10,color=MUTED)
    box(ax,57,21,40,12,choose('融合：英雄 32D + 地图 64D + 上下文 32D\n128 → 128 → 128（ReLU）','Fusion: hero 32D + map 64D + context 32D\n128 → 128 → 128 (ReLU)',lang),BLUE,11)
    arrow(ax,(77,42.5),(77,33.5))
    arrow(ax,(48.5,10),(56.5,26.5),BLUE)
    box(ax,57,3,19,11,choose('Actor\n16 个动作 logits','Actor\n16 action logits',lang),TEAL,11)
    box(ax,79,3,18,11,choose('Critic\n1 个状态价值','Critic\n1 state value',lang),GOLD,11)
    arrow(ax,(67,20.5),(67,14.5)); arrow(ax,(88,20.5),(88,14.5))
    fig.text(.04,.025,choose('来源：agent_ppo/model/model.py；参数数量由权重与网络检查生成。','Source: agent_ppo/model/model.py; parameter count from checkpoint inspection.',lang),fontsize=10,color=MUTED)
    save(fig,'architecture',lang)


def workflow(lang):
    fig,ax = canvas(choose('采样、优化与独立验证','Sampling, optimization and held-out validation',lang),choose('图示为现有代码的配置流程；历史训练曲线和验证日志未留存','Configured workflow in the archived code; historical training curves and validation logs are unavailable',lang),9)
    box(ax,1,62,22,18,choose('训练环境\n地图 1-8，按顺序轮换\n10 宝箱 / 2 BUFF / 1000 步','Training environment\nMaps 1-8, sequential rotation\n10 treasures / 2 buffs / 1000 steps',lang),TEAL,11)
    box(ax,29,62,23,18,choose('Actor 与预处理器\n特征、地图、BFS、记忆\n合法动作上的随机采样','Actor + preprocessor\nFeatures, map, BFS and memory\nSample over legal actions',lang),BLUE,11)
    box(ax,59,62,18,18,choose('整局轨迹\nGAE：γ=0.99\nλ=0.95','Episode trajectory\nGAE: γ=0.99\nλ=0.95',lang),TEAL,11)
    box(ax,83,62,16,18,choose('样本池\n容量 2048\n批次 512','Replay pool\nCapacity 2048\nBatch 512',lang),BLUE,11)
    arrow(ax,(23.5,71),(28.5,71)); arrow(ax,(52.5,71),(58.5,71)); arrow(ax,(77.5,71),(82.5,71))
    box(ax,72,32,27,18,choose('Learner：PPO 更新\n4 轮 / mini-batch 256\n策略剪切 + 价值剪切 + 熵','Learner: PPO updates\n4 epochs / mini-batch 256\nPolicy clip + value clip + entropy',lang),BLUE,11)
    arrow(ax,(91,61.5),(91,50.5))
    box(ax,38,32,25,18,choose('模型同步与保存\n模型池同步：1 分钟\nworkflow 保存：1800 秒','Model synchronization / save\nModel-pool sync: 1 minute\nWorkflow save: 1800 seconds',lang),TEAL,11)
    arrow(ax,(71.5,41),(63.5,41)); arrow(ax,(45,50.5),(41,61.5))
    box(ax,1,10,27,22,choose('独立验证环境\n地图 9、10\n每轮共 4 局 / 间隔 600 秒\n使用当前模型的最大概率动作','Held-out validation\nMaps 9 and 10\n4 episodes total / 600-second interval\nGreedy actions from current weights',lang),GOLD,10.5)
    box(ax,38,10,35,13,choose('val_* 指标单独上报\n不发送训练样本、不做梯度更新','Report val_* metrics separately\nNo training samples or gradient updates',lang),GOLD,11)
    arrow(ax,(28.5,21),(37.5,17),GOLD)
    ax.text(76,14,choose('首次验证可立即触发\n不是每张图各 4 局','First round can start immediately\n4 episodes total, not per map',lang),fontsize=10.5,color=MUTED)
    fig.text(.04,.02,choose('原始轻量验证会重复更新同帧预处理状态，具体语义见技术说明；历史验证日志未留存。','Lightweight validation repeats same-frame state updates; see technical details. Historical validation logs are unavailable.',lang),fontsize=10,color=MUTED)
    save(fig,'workflow',lang)


def profiles(lang):
    fig = plt.figure(figsize=(16,8),facecolor='white')
    fig.text(.06,.94,choose('输入组成与模型参数分布','Input composition and parameter allocation',lang),fontsize=23,color=INK,fontweight='bold')
    fig.text(.06,.88,choose('输入维度来自配置；参数计数来自严格匹配的检查点。两种单位不能直接比较。','Input dimensions come from configuration; parameter counts come from the matched checkpoint. These are different units.',lang),fontsize=11,color=MUTED)
    ax = fig.add_axes([.08,.2,.38,.61])
    labels = ['英雄 / Hero','怪物 / Monsters','宝箱 / Treasures','BUFF','拓扑 / Topology','记忆 / Memory'] if lang=='zh' else ['Hero','Monsters','Treasures','Buffs','Topology','Memory']
    dims = [14,16,14,8,13,8]
    y = np.arange(6); ax.barh(y,dims,color=TEAL,height=.58); ax.set_yticks(y,labels); ax.invert_yaxis();ax.set_xlim(0,20)
    ax.set_xlabel(choose('结构化特征维度：共 73','Structured feature dimensions: 73 total',lang),color=MUTED)
    for yi,v in enumerate(dims):ax.text(v+.35,yi,str(v),va='center',fontsize=12,color=INK)
    ax.set_title(choose('地图另占 3528 维（8 × 21 × 21）','Map adds 3528 dimensions (8 × 21 × 21)',lang),fontsize=12,color=INK,pad=15)
    ax2 = fig.add_axes([.61,.2,.33,.61])
    groups = CHECKPOINT['parameters_by_module']
    names = ['Map CNN','6 MLP encoders','Type gate','Fusion','Actor + critic']
    vals = [groups['map_encoder'],sum(v for k,v in groups.items() if k.endswith('_encoder') and k!='map_encoder'),groups['type_gate'],groups['fusion'],groups['actor_head']+groups['critic_head']]
    ax2.barh(np.arange(5),vals,color=[BLUE,TEAL,GOLD,BLUE,TEAL],height=.6);ax2.set_yticks(np.arange(5),names);ax2.invert_yaxis();ax2.set_xlim(0,max(vals)*1.28)
    for yi,v in enumerate(vals):ax2.text(v+500,yi,f'{v:,}',va='center',fontsize=11,color=INK)
    ax2.set_xlabel(choose('可训练参数数量：共 75,814','Trainable parameters: 75,814 total',lang),color=MUTED)
    for a in [ax,ax2]:
        a.spines[['top','right','left']].set_visible(False);a.spines['bottom'].set_color(LINE);a.tick_params(axis='both',colors=MUTED);a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
    fig.text(.06,.06,choose('总观测为 3601 维；地图维度占 97.97%。高维地图输入经过 CNN 压缩为 64 维嵌入。','Total observation: 3601D; map dimensions account for 97.97%. CNN compresses the map to a 64D embedding.',lang),fontsize=11,color=MUTED)
    save(fig,'profiles',lang)


def rewards(lang):
    labels = ['拾取 1 个宝箱','拾取 BUFF','成功高危闪现','安全宝箱闪现','无效移动','安全时磨蹭','高危无效闪现','存活至步数上限','被怪物捕获','异常截断'] if lang=='zh' else ['Collect one treasure','Collect a buff','Effective high-risk flash','Safe treasure flash','Invalid move','Loiter in safety','Ineffective high-risk flash','Reach step limit','Captured by monster','Abnormal truncation']
    vals = [Config.REWARD_TREASURE_GAIN_SCALE,Config.REWARD_BUFF_PICK,Config.REWARD_FLASH_HIGH_GOOD,Config.REWARD_FLASH_TREASURE_SAFE,Config.REWARD_INVALID_MOVE,Config.REWARD_LOITER,Config.REWARD_FLASH_HIGH_BAD,Config.REWARD_COMPLETED,Config.REWARD_TERMINATED,Config.REWARD_ABNORMAL]
    fig,ax = plt.subplots(figsize=(16,8.5),facecolor='white');fig.subplots_adjust(left=.25,right=.91,top=.79,bottom=.19)
    fig.text(.06,.94,choose('奖励塑形：条件事件的增量','Reward shaping: conditional event increments',lang),fontsize=23,color=INK,fontweight='bold')
    fig.text(.06,.87,choose('配置值图示，不是实测回报；同一步可叠加多个奖励项','Configured increments, not measured returns; multiple terms may apply to the same step',lang),fontsize=11,color=MUTED)
    y = np.arange(len(vals));ax.barh(y,vals,color=[GREEN if v>0 else RED for v in vals],height=.61);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(-2.4,1.6);ax.axvline(0,color=LINE,linewidth=1)
    for yi,v in enumerate(vals):ax.text(v+(.05 if v>=0 else -.05),yi,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=11,color=INK)
    ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color(LINE);ax.tick_params(colors=MUTED);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.set_xlabel(choose('事件奖励增量（训练用，无得分单位）','Event reward increment (training quantity, not score points)',lang),color=MUTED)
    fig.text(.06,.09,choose('步数项：0.02 × Δ步数得分；按计分规则每普通存活步约 +0.03。安全、危险、夹击及地形项另计。','Step term: 0.02 × Δstep_score, about +0.03 per ordinary surviving step. Safety, danger, squeeze and terrain terms are additional.',lang),fontsize=10.5,color=MUTED)
    fig.text(.06,.045,choose('来源：agent_ppo/conf/conf.py 与 preprocessor.py；闪现奖励依据风险和脱险质量条件启用。','Source: agent_ppo/conf/conf.py and preprocessor.py; flash increments depend on risk and escape quality.',lang),fontsize=10.5,color=MUTED)
    save(fig,'rewards',lang)


if __name__ == '__main__':
    for lang in ['zh','en']:
        for function in [results,architecture,workflow,profiles,rewards]:
            function(lang)
    print('Generated 10 bilingual figures in SVG and PNG formats.')
