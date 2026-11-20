import {state} from '../stores/session.js';
import {esc} from '../utils/html.js';
import {table} from '../components/common/table.js';
import {badge} from '../components/common/status.js';
import {field,select} from '../components/common/fields.js';
import {ANNOTATOR_STATUS_LABELS,CLAIM_STATUS_LABELS} from '../constants/enums.js';

const pc=v=>v===null||v===undefined?'—':(100*v).toFixed(1)+'%';
const num=v=>v===null||v===undefined?'—':Number(v).toFixed(3);
const claimOptions=()=>state.claims.map(c=>[c.id,`${c.id} · ${c.code}`]);

function opinionRow(c){
  const ops=state.opinions.filter(o=>o.claim_id===c.id);
  const gold=state.gold.find(g=>g.claim_id===c.id);
  const shown=ops.map(o=>`${esc(o.annotator)}:${o.label}`).join(' · ')||'无独立意见';
  return [c.id,c.code,shown,gold?(gold.label?'金标准 异常':'金标准 正常'):'—'];
}

function annotatorTable(result){
  if(!result?.annotators?.length)return '<p>尚无标注者估计。</p>';
  return table(['财务','意见','重叠','漏报率','误报率','准确率','区间','判断'],
    result.annotators.map(a=>[a.annotator,a.opinions,a.overlap,pc(a.false_negative_rate),pc(a.false_positive_rate),
      pc(a.accuracy),a.sensitivity_ci?`${pc(a.sensitivity)} [${pc(a.sensitivity_ci[0])}, ${pc(a.sensitivity_ci[1])}]`:'—',
      `${ANNOTATOR_STATUS_LABELS[a.status]||a.status}${a.note?' · '+a.note:''}`]));
}

function claimTable(result){
  const rows=(result?.claims||[]).slice(0,120);
  return table(['票据','意见数','多数票','异常后验','区间','状态'],
    rows.map(c=>[`#${c.claim_id} ${c.claim_code}`,c.opinions,c.majority_label??'平票',
      pc(c.probability),c.low!==null&&c.low!==undefined?`[${pc(c.low)}, ${pc(c.high)}]`:'—',
      `${CLAIM_STATUS_LABELS[c.status]||c.status}${c.gold?'（金标准）':''}`]));
}

function evidenceBlock(result){
  if(!result)return '';
  const e=result.evidence||{};
  return `<div class="eval-grid"><div><h3>校准依据</h3>
    <p>参与估计：${e.claims} 笔票据 · ${e.opinions} 条意见 · ${e.annotators} 位财务 · ${e.gold} 笔金标准（共 ${e.gold_total} 笔）。</p>
    ${(result.checks||[]).map(c=>`<div class="coef"><span style="width:auto">${c.ok?'✓':'✗'} ${esc(c.detail)}</span></div>`).join('')}
    <p class="notice">多数票与模型不一致 ${result.majority_disagreements} 笔。这里把多数意见当对照，不当真值。</p></div>
    <div><h3>互不重叠人群</h3>
    ${(e.groups||[]).map((g,i)=>`<p>人群 ${i+1}：${g.claims} 笔票据 / ${g.opinions} 条意见 · 标注者 ${g.annotators.join('、')||'—'} · 金标准锚点 ${g.gold_anchors}</p>`).join('')}
    ${(result.excluded_groups||[]).length?`<p class="notice">未纳入估计：${result.excluded_groups.length} 个缺少金标准锚点的人群，其票据不给概率。</p>`:''}
    <p>潜变量患病率 ${pc(result.prevalence)} · EM 迭代 ${result.iterations} · 自助法有效 ${result.bootstrap_valid} 次。</p></div></div>`;
}

function calibrationBody(t){
  if(!t)return '<p class="notice">选择或创建一个校准批次，查看各财务的漏报、误报率与逐票后验。</p>';
  if(t.state==='queued'||t.state==='running')return `<p class="notice">批次 #${t.id} 正在${t.state==='queued'?'等待估计':'估计中'}，完成后刷新查看。意见与金标准已冻结在提交时的版本。</p>`;
  if(t.state==='failed'){
    const reasons=t.result?.reasons||[];
    return `<p class="notice">本批次拒绝给出估计：${esc(t.error||'执行失败')}</p>
      ${reasons.length?`<ul>${reasons.map(r=>`<li>${esc(r)}</li>`).join('')}</ul>`:''}
      ${(t.result?.checks||[]).length?`<p>${t.result.checks.map(c=>`${c.ok?'✓':'✗'} ${esc(c.detail)}`).join('；')}</p>`:''}
      <p>不会用多数票冒充真值，也不会输出虚假的高置信度。补充独立意见或确认少量金标准后重试；重试沿用同一批意见。</p>`;
  }
  const r=t.result;
  return `<p class="eyebrow">方法 ${r.method} · seed ${r.seed} · 快照 ${esc(t.snapshot_hash?.slice(0,12)||'')}</p>
    ${evidenceBlock(r)}
    <h3>各财务漏报 / 误报估计</h3>${annotatorTable(r)}
    <h3>逐票异常后验</h3>${claimTable(r)}
    <p class="notice">${esc(r.warning)}</p>`;
}

export function render(){
  const t=state.selectedCalibration;
  return `<div class="split"><section class="panel">
    <h2>标注可靠性校准</h2>
    <p>历史人工标签可能把多人共同误判当成真相。这里用不完整重叠的独立意见与少量已确认金标准，估计各人漏报、误报率与逐票异常概率，不把多数意见直接当真值。</p>
    <h3>校准批次</h3>
    ${state.calibrations.map(c=>`<button class="study" data-calibration="${c.id}">#${c.id} ${esc(c.name)} · 尝试${c.attempts} ${badge(c.state)}</button>`).join('')||'<p>暂无批次。录入独立意见后创建。</p>'}
    ${calibrationBody(t)}
    ${t&&(t.state==='queued'||t.state==='running')?`<button id="refresh-calibration">刷新批次状态</button>`:''}
    ${t?.state==='failed'?`<button id="retry-calibration" data-id="${t.id}">沿用同批意见重试</button>`:''}
    ${t?.state==='done'?`<button id="calibration-export" data-id="${t.id}">下载校准依据 JSON</button>`:''}
    <h3>票据与独立意见</h3>${table(['ID','报销编号','独立意见（标签）','金标准'],state.claims.slice(0,80).map(opinionRow))}
  </section><aside class="side">
    <h2>录入独立意见</h2>
    <form id="opinion">${select('票据','claim_id',claimOptions())}${field('财务姓名','annotator','text','required maxlength=80 placeholder="财务-沈"')}${select('独立判断','label',[['0','正常'],['1','异常']])}<label>备注<textarea name="note" rows="2" maxlength="1000"></textarea></label><button class="primary">保存意见</button></form>
    <p>同一位财务对同一票据只保留最新一版，每次更正都会追加修订记录。</p>
    <h2>确认金标准</h2>
    <form id="gold">${select('票据','claim_id',claimOptions())}${select('已核实结论','label',[['0','正常'],['1','异常']])}<label>核实依据<textarea name="note" minlength="4" required rows="3">已核对原始票据与审批流</textarea></label><button class="primary">确认为金标准</button></form>
    <p>金标准仅由负责人确认，用于锚定方向与评估标注者，不参与留出评估。</p>
    <h2>创建校准批次</h2>
    <form id="calibration">${field('批次名称','name','text','required value="11月标注校准"')}<button class="primary">冻结意见并校准</button></form>
  </aside></div>`;
}
