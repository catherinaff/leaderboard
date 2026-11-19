import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const root=fileURLToPath(new URL('..',import.meta.url));
const read=name=>fs.readFileSync(path.join(root,'site',name),'utf8');
async function boot(prefix,search='') {
  const page='https://official.example'+prefix+'/';
  const elements=new Map();
  function get(id) {
    if(!elements.has(id)) {
      const events=new Map();
      elements.set(id,{innerHTML:'',textContent:'',value:'',hidden:false,disabled:false,
        querySelectorAll(){return [];},addEventListener(n,f){events.set(n,f);},
        dispatch(n,e){events.get(n)?.(e);},showModal(){this.open=true;},close(){this.open=false;}});
    }
    return elements.get(id);
  }
  const requests=[];
  const location={href:page+search,origin:'https://official.example',search};
  const env={URL,URLSearchParams,console,location,
    document:{baseURI:page,currentScript:{src:page+'app-path.js'},getElementById:get},
    history:{replaceState(a,b,u){location.href=String(u);}},window:{addEventListener(){}},
    fetch:async value=>{
      const u=new URL(value,page);requests.push(u.pathname);
      assert.ok(u.pathname.startsWith(prefix+'/'),u.pathname);
      const relative=u.pathname.slice(prefix.length+1);
      const file=path.join(root,'site',relative);
      return {ok:fs.existsSync(file),json:async()=>JSON.parse(fs.readFileSync(file,'utf8'))};
    }};
  vm.runInNewContext(read('app-path.js')+'\n'+read('ranking-model.js')+'\n'+read('ranking.js'),env);
  await new Promise(r=>setImmediate(r));
  return {get,env,requests};
}

const expectedMethods=['mosaic','onehar','deepconvlstm','if_convtransformer','onellm','imu2clip','sensorllm','gpt4ts','timellm','units','calf','hargpt','bymyeyes','imugpt','ispl_inception','hart','mobilehart','unimts','limu_bert'];
const paperData=()=>JSON.parse(read('data/paper_transfer_evidence.json'));
const measuredData=()=>JSON.parse(read('data/measured_results_20261119.json'));

test('all scoped rankings and data requests work at root and arbitrary subpaths',async()=>{
  for(const prefix of ['', '/leaderboard','/research/leaderboard']) {
    const app=await boot(prefix);
    const html=app.get('leaderboard-wrap').innerHTML;
    assert.match(html,/measured-20261119:onehar:uci_har/);
    assert.match(html,/CDNet adapter\*/);
    assert.match(html,/One Model to Fit Them All \(oneHAR\)/);
    for(const id of expectedMethods) assert.ok(html.includes('data-method="'+id+'"'),id);
    for(const id of ['robusthar','tinyhar','attend_and_discriminate','deepsense']) assert.ok(!html.includes('data-method="'+id+'"'),id);
    assert.equal((html.match(/data-method=/g)||[]).length,19);
    assert.equal(app.requests.length,6);
    assert.equal(app.env.IoTBenchPath('/downloads/a.zip'),'https://official.example'+prefix+'/downloads/a.zip');
    assert.equal(app.env.IoTBenchPath('data/a.json'),'https://official.example'+prefix+'/data/a.json');
    assert.equal(app.env.IoTBenchPath('https://paper.example/paper'),'https://paper.example/paper');
  }
});
test('every legacy overall route resolves to the same four-dataset measured-first ranking',async()=>{
  for(const query of ['', '?board=imu&view=overall', '?board=imu&tab=cross-dataset&view=overall&aggregate=onehar-lodo', '?board=imu&view=overall&tab=standard', '?board=imu&view=overview']) {
    const app=await boot('/leaderboard',query);
    assert.match(app.get('result-summary').textContent,/19 ranked models · 4 datasets/);
    const html=app.get('leaderboard-wrap').innerHTML;
    assert.match(html,/UCI-HAR · 6 classes/);assert.match(html,/RealWorld · 8 classes/);
    assert.match(html,/68\.43/);assert.match(html,/comparison-asterisk/);
    assert.match(html,/PAMAP2<sup/);assert.match(html,/USC-HAD<sup/);
    assert.match(html,/pamap2_12c|measured-20261119:onehar:pamap2/);
    assert.match(html,/usc_had_12c|measured-20261119:onehar:usc_had/);
    assert.doesNotMatch(html,/Partial coverage|unranked|available-only/);
    assert.equal(app.get('reference-footnote').hidden,false);
    assert.equal(app.get('mode-controls').hidden,true);
    assert.equal(app.get('protocol-field').hidden,true);
  }
  assert.match(read('index.html'),/各占 25%/);
  assert.match(read('index.html'),/监督训练、LODO、zero-shot、划分和标注预算/);
  assert.doesNotMatch(read('index.html'),/各方法按其可用数据集|Paper-reported averages/);
  assert.equal(read('index.html'),read('leaderboard.html'));
});
test('scope preserves every paper comparison and distinct class variant',async()=>{
  const app=await boot('/leaderboard'),M=app.env.IoTBenchRanking,d=paperData();
  const scope=M.paperScope(d);
  assert.equal(scope.catalog.length,21);assert.equal(scope.familyCount,19);assert.equal(scope.methods.length,19);
  assert.equal(scope.observations.length,571);
  assert.deepEqual([...scope.methods].map(m=>m.id).sort(),expectedMethods.toSorted());
  assert.deepEqual([...scope.common],['uci_har_6c','realworld_8c']);
  for(const id of ['pamap2_9c','pamap2_12c','usc_had_7c','usc_had_12c','wisdm_18c','opportunity_4c']) assert.ok(scope.catalog.some(d=>d.id===id));
  assert.ok(!scope.catalog.some(d=>['wisdm','opportunity_a'].includes(d.id)));
  for(const [id,n] of [['mosaic-zero-shot',10],['mosaic-few-shot',7],['mosaic-full-shot',7],['onehar-lodo',8],['onehar-joint',7]]) assert.equal(d.comparisons[id].length,n,id);
});
test('overall means share a fixed denominator and use one documented setting per model',async()=>{
  const app=await boot('/leaderboard'),M=app.env.IoTBenchRanking,d=paperData(),before=JSON.stringify(d);
  const b=M.paperBenchmark(d);
  assert.equal(b.entries.length,19);
  for(const e of b.entries){
    assert.equal(e.coverage,4);
    assert.deepEqual([...e.datasetIds],['uci_har','realworld','pamap2','usc_had']);
    assert.deepEqual([...e.values].map(v=>v.comparisonDataset),['uci_har','realworld','pamap2','usc_had']);
    assert.equal(new Set(e.labelSets).size,4);
    assert.deepEqual([...e.values].map(v=>v.classMismatch),[false,false,true,true]);
    assert.ok(e.values.every(v=>v.protocol===e.protocol&&v.source==='paper'));
    assert.equal(e.mean,e.values.reduce((sum,v)=>sum+v.value,0)/4);
    assert.equal(e.sourceRuns,0);
  }
  const one=b.entries.find(e=>e.method.id==='onehar'),mosaic=b.entries.find(e=>e.method.id==='mosaic');
  assert.ok(Math.abs(one.mean-.9205)<1e-12);assert.ok(Math.abs(mosaic.mean-.68425)<1e-12);
  assert.equal(one.protocol,'onehar-joint');assert.equal(mosaic.protocol,'mosaic-full-shot');
  assert.equal(b.entries.find(e=>e.method.id==='sensorllm').protocol,'mosaic-full-shot');
  assert.match(b.entries.find(e=>e.method.id==='limu_bert').protocolLabel,/per-dataset/);
  assert.deepEqual([...one.labelSets],['uci_har_6c','realworld_8c','pamap2_9c','usc_had_7c']);
  assert.deepEqual([...mosaic.labelSets],['uci_har_6c','realworld_8c','pamap2_12c','usc_had_12c']);
  assert.equal(JSON.stringify(d),before);
  assert.ok(b.entries.every((e,i)=>i===0||b.entries[i-1].mean>=e.mean));
});
test('selection is independent of score magnitude, retains zeros, and never fills missing targets',async()=>{
  const app=await boot('/leaderboard'),M=app.env.IoTBenchRanking,d=paperData();
  const full=d.comparisons['mosaic-full-shot'].find(e=>e.method_id==='sensorllm');
  const zero=d.comparisons['mosaic-zero-shot'].find(e=>e.method_id==='sensorllm');
  full.results.forEach(r=>r.accuracy=0);zero.results.forEach(r=>r.accuracy=100);
  const b=M.paperBenchmark(d),e=b.entries.find(e=>e.method.id==='sensorllm');
  assert.equal(e.protocol,'mosaic-full-shot');assert.equal(e.mean,0);assert.equal(e.coverage,4);
  assert.equal(M.paperScope(d).observations.find(r=>r.id==='transfer:mosaic-zero-shot:gpt4ts:uci_har_6c').value,0);
  for(const g of d.groups)g.results=g.results.filter(r=>r.target_id!=='realworld_8c'||g.method_id!=='onehar');
  // Removing one method's target cannot create a missing=zero result.
  const custom=M.paperBenchmark(d,['uci_har_6c','realworld_8c']);
  assert.ok(!custom.entries.some(e=>e.method.id==='onehar'));
});
test('reversing display order retains true ranks',async()=>{
  const app=await boot('/leaderboard');
  const before=app.env.IoTBenchRanking.currentBenchmark(paperData(),measuredData());
  const leading=before.entries.find(e=>e.rank===1).method.id;
  app.get('sort-results').dispatch('click');
  const html=app.get('leaderboard-wrap').innerHTML;
  assert.ok(html.indexOf('data-method="onellm"')<html.indexOf('data-method="onehar"'));
  assert.match(html,/<tr class="reference-row" data-method="onellm"><td><span class="rank-number">19<\/span>/);
  assert.ok(html.includes(`<tr class="reference-row leading" data-method="${leading}"><td><span class="rank-number">1</span>`));
});
test('overall Details identify the four selected targets and their actual source',async()=>{
  const app=await boot('/leaderboard');
  app.get('leaderboard-wrap').dispatch('click',{target:{closest:s=>s==='[data-summary]'?{dataset:{summary:'onehar'}}:null}});
  assert.match(app.get('detail-context').textContent,/4 datasets/);
  const runs=measuredData().records.filter(r=>r.method==='onehar').length;
  assert.ok(app.get('detail-body').innerHTML.includes(`${runs} measured results and ${4-runs} paper results`));
  assert.equal((app.get('detail-body').innerHTML.match(/data-result=/g)||[]).length,4);
  assert.match(app.get('detail-body').innerHTML,/CDNet|Our run/);
  app.get('leaderboard-wrap').dispatch('click',{target:{closest:s=>s==='[data-summary]'?{dataset:{summary:'mosaic'}}:null}});
  app.get('detail-body').dispatch('click',{target:{closest:()=>({dataset:{result:'transfer:mosaic-full-shot:mosaic:uci_har_6c'}})}});
  assert.match(app.get('detail-body').innerHTML,/Table 4 · PDF p\.16/);
  assert.match(app.get('detail-body').innerHTML,/https:\/\/dl\.acm\.org\/doi\/10\.1145\/3831643/);
});
test('dataset views use four families and retain starred class variants and measured sources',async()=>{
  const app=await boot('/leaderboard','?board=imu&view=ranking&dataset=pamap2_9c');
  assert.equal(app.get('dataset-select').value,'pamap2');
  assert.equal((app.get('dataset-select').innerHTML.match(/<option /g)||[]).length,4);
  assert.equal(app.get('dataset-heading').textContent,'PAMAP2');
  assert.match(app.get('leaderboard-wrap').innerHTML,/oneHAR/);
  if(!measuredData().records.some(r=>r.method==='onehar'&&r.dataset==='pamap2'))
    assert.match(app.get('leaderboard-wrap').innerHTML,/PAMAP2 · 9 classes/);
  assert.match(app.get('leaderboard-wrap').innerHTML,/PAMAP2 · 12 classes/);
  assert.match(app.get('leaderboard-wrap').innerHTML,/MOSAIC|Our run/);
  assert.doesNotMatch(app.get('leaderboard-wrap').innerHTML,/TinyHAR|RobustHAR/);
  const full=await boot('/leaderboard','?view=ranking&dataset=wisdm_18c&protocol=mosaic-full-shot');
  assert.equal(full.get('dataset-select').value,'uci_har');
  assert.match(full.get('result-summary').textContent,/19 ranked models · Accuracy · measured first/);
  assert.match(full.get('leaderboard-wrap').innerHTML,/SensorLLM|GPT4TS/);
});

test('measured overlay uses all verified results regardless of gap and leaves missing runs as paper',async()=>{
  const app=await boot('/leaderboard'),M=app.env.IoTBenchRanking,p=paperData(),d=measuredData();
  const before=JSON.stringify([p,d]),b=M.currentBenchmark(p,d);
  assert.equal(b.entries.length,19);
  const added=d.records.filter(r=>r.method==='onehar');
  assert.equal(b.observations.filter(r=>r.source==='run').length,d.records.length);
  assert.equal(b.observations.filter(r=>r.source==='paper').length,76-d.records.length);
  assert.equal(b.observations.filter(r=>r.provisional).length,d.records.filter(r=>r.status==='provisional_complete_seeds').length);
  assert.ok(d.records.filter(r=>r.method!=='onehar').reduce((sum,r)=>sum+r.completed_folds,0)>=2163);
  assert.equal(d.totals.verified_fold_records,d.records.reduce((sum,r)=>sum+r.completed_folds,0));
  for(const e of b.entries){
    assert.equal(e.values.length,4);
    assert.deepEqual([...e.values].map(r=>r.dataset),['uci_har','realworld','pamap2','usc_had']);
    assert.equal(e.mean,e.values.reduce((sum,r)=>sum+r.value,0)/4);
  }
  for(const raw of d.records){
    const selected=b.observations.find(r=>r.method.id===raw.method&&r.dataset===raw.family);
    assert.equal(selected.source,'run');
    assert.equal(selected.value,raw.metrics.accuracy);
    assert.equal(selected.sourceRecord.id,raw.id);
  }
  const sensor=b.entries.find(e=>e.method.id==='sensorllm');
  const sensorRuns=d.records.filter(r=>r.method==='sensorllm').length;
  assert.equal(sensor.sourceRuns,sensorRuns);assert.equal(sensor.sourcePapers,4-sensorRuns);
  assert.equal(sensor.values[0].source,'run');
  assert.equal(sensor.values[0].value,d.records.find(r=>r.method==='sensorllm'&&r.family==='uci_har').metrics.accuracy);
  assert.notEqual(sensor.values[0].value,sensor.values[0].paperReference.value);
  const one=b.entries.find(e=>e.method.id==='onehar');
  const measuredOne=new Set(added.map(r=>r.family));
  assert.ok(one.values.every(r=>measuredOne.has(r.dataset)?r.source==='run':r.paperFallback));
  assert.equal(one.values[0].source,'run');
  const rw=b.observations.find(r=>r.method.id==='if_convtransformer'&&r.dataset==='realworld');
  assert.notEqual(rw.value,rw.metrics.f1_macro);
  assert.equal(rw.value,rw.metrics.accuracy);
  assert.equal(JSON.stringify([p,d]),before);
});

test('unfinished subject sets, unverified and smoke results cannot replace paper values',async()=>{
  const app=await boot('/leaderboard'),M=app.env.IoTBenchRanking,p=paperData(),data=measuredData();
  const full=data.records.find(r=>r.method==='calf'&&r.family==='realworld');
  const paper=M.currentBenchmark(p,{records:[]}).entries.find(e=>e.method.id==='calf').values[1];
  for(const change of [{completed_folds:29},{complete_subjects_per_seed:14},{verified:false},{status:'smoke'},{seeds:[42,42]},{metrics:{accuracy:1.1}}]){
    const bad={...structuredClone(full),...change};
    const selected=M.currentBenchmark(p,{records:[bad]}).entries.find(e=>e.method.id==='calf').values[1];
    assert.equal(selected.source,'paper');assert.equal(selected.value,paper.value);
  }
  assert.throws(()=>M.currentBenchmark(p,{records:[full,full]}),/Duplicate measured aggregate/);
});

test('provisional results cover all subjects and zero-shot results are one pass without fabricated SD',async()=>{
  const app=await boot('/leaderboard'),M=app.env.IoTBenchRanking,d=measuredData(),b=M.currentBenchmark(paperData(),d);
  for(const r of b.observations.filter(r=>r.provisional&&r.method.id!=='onehar')){
    const subjects={uci_har:1,pamap2:8,usc_had:14,realworld:15}[r.dataset];
    assert.equal(r.sourceRecord.complete_subjects_per_seed,subjects);
    assert.equal(r.sourceRecord.completed_folds,subjects*r.sourceRecord.seeds.length);
    assert.match(r.sourceLabel,/provisional/);
    assert.ok(r.sourceRecord.seeds.length>=1&&r.sourceRecord.seeds.length<5);
  }
  for(const r of b.observations.filter(r=>r.method.id==='onellm')){
    assert.equal(r.runKind,'zero_shot');assert.equal(r.sourceRecord.seeds.length,1);
    assert.equal(r.uncertainty,undefined);
  }
  app.get('leaderboard-wrap').dispatch('click',{target:{closest:s=>s==='[data-result]'?{dataset:{result:'measured-20261119:calf:realworld_har'}}:null}});
  assert.match(app.get('detail-body').innerHTML,/\* Provisional/);
  assert.match(app.get('detail-body').innerHTML,/42, 123 \(2\/5\)/);
  assert.match(app.get('detail-body').innerHTML,/30 \/ 75/);
  assert.match(app.get('detail-body').innerHTML,/Verified results &amp; fold evidence/);
});

test('oneHAR additions require complete canonical seed sets and disclose the supervised adapter',async()=>{
  const app=await boot('/leaderboard'),M=app.env.IoTBenchRanking,d=measuredData();
  const one=M.currentBenchmark(paperData(),d).entries.find(e=>e.method.id==='onehar');
  for(const r of one.values.filter(v=>v.source==='run')){
    assert.ok(['uci_har','pamap2','usc_had','realworld'].includes(r.dataset));
    const expected={uci_har:1,pamap2:8,usc_had:14,realworld:15}[r.dataset];
    assert.equal(r.sourceRecord.completed_folds,r.sourceRecord.seeds.length*expected);
    assert.equal(r.sourceRecord.num_classes,{uci_har:6,pamap2:12,usc_had:12,realworld:8}[r.dataset]);
    assert.match(r.variantLabel,/CDNet supervised adapter/);
    assert.equal(r.provisional,r.sourceRecord.seeds.length<5);
  }
  app.get('leaderboard-wrap').dispatch('click',{target:{closest:s=>s==='[data-result]'?{dataset:{result:'measured-20261119:onehar:uci_har'}}:null}});
  assert.match(app.get('detail-body').innerHTML,/CDNet supervised adapter/);
  assert.match(app.get('detail-body').innerHTML,/no LLM synthesis/);
  assert.match(read('index.html'),/oneHAR \/ CDNet 实测/);
});
test('retired Sensor QA links resolve to IMU without loading QA data',async()=>{
  for(const search of ['?board=qa&view=overall','?board=qa&view=ranking&dataset=quants_binary']){
    const app=await boot('/leaderboard',search);
    assert.doesNotMatch(app.get('leaderboard-wrap').innerHTML,/Results could not be loaded|No matching results/);
    assert.match(app.env.location.href,/board=imu/);
    assert.match(app.get('leaderboard-wrap').innerHTML,/MOSAIC/);
    assert.ok(!app.requests.some(p=>p.includes('sensor_qa')));
  }
  for(const file of ['index.html','leaderboard.html'])assert.doesNotMatch(read(file),/data-board="qa"|Sensor QA|sensor question answering/);
});
test('Pages submissions keep the authorized server upload portal',()=>{
 const html=read('submit-model.html');assert.match(html,/http:\/\/10\.130\.151\.16:40677\/submit-model\.html/);
 assert.match(html,/campus network or VPN/);assert.doesNotMatch(html,/<form|api\/submissions/);
});
test('existing download links remain official and distinct from paper-specific class variants',()=>{
 const data=JSON.parse(read('data/datasets.json')),html=read('datasets.html');assert.equal(data.datasets.length,5);
 for(const entry of data.datasets){assert.equal(entry.archive_url,entry.official_archive_url);assert.match(entry.archive_url,/^https:/);assert.ok(html.includes('href="'+entry.official_archive_url+'"'));}
 assert.match(html,/checksums describe the mirror snapshot/);
});

test('Tasks and Datasets retain their original task cards and download layout',()=>{
 assert.match(read('tasks.html'),/card-grid/);assert.match(read('tasks.html'),/Loading tasks/);
 assert.match(read('datasets.html'),/Download the core datasets/);
 for(const file of ['tasks.html','datasets.html'])assert.doesNotMatch(read(file),/paper-catalog/);
});
