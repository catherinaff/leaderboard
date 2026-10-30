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

test('all scoped rankings and data requests work at root and arbitrary subpaths',async()=>{
  for(const prefix of ['', '/leaderboard','/research/leaderboard']) {
    const app=await boot(prefix);
    const html=app.get('leaderboard-wrap').innerHTML;
    assert.match(html,/92\.05/);
    assert.match(html,/One Model to Fit Them All \(oneHAR\)/);
    for(const id of expectedMethods) assert.ok(html.includes('data-method="'+id+'"'),id);
    for(const id of ['robusthar','tinyhar','attend_and_discriminate','deepsense']) assert.ok(!html.includes('data-method="'+id+'"'),id);
    assert.equal((html.match(/data-method=/g)||[]).length,19);
    assert.equal(app.requests.length,5);
    assert.equal(app.env.IoTBenchPath('/downloads/a.zip'),'https://official.example'+prefix+'/downloads/a.zip');
    assert.equal(app.env.IoTBenchPath('data/a.json'),'https://official.example'+prefix+'/data/a.json');
    assert.equal(app.env.IoTBenchPath('https://paper.example/paper'),'https://paper.example/paper');
  }
});
test('every legacy overall route resolves to the same four-dataset paper ranking',async()=>{
  for(const query of ['', '?board=imu&view=overall', '?board=imu&tab=cross-dataset&view=overall&aggregate=onehar-lodo', '?board=imu&view=overall&tab=standard', '?board=imu&view=overview']) {
    const app=await boot('/leaderboard',query);
    assert.match(app.get('result-summary').textContent,/19 ranked models · 4 datasets/);
    const html=app.get('leaderboard-wrap').innerHTML;
    assert.match(html,/UCI-HAR · 6 classes/);assert.match(html,/RealWorld · 8 classes/);
    assert.match(html,/68\.43/);assert.match(html,/comparison-asterisk/);
    assert.match(html,/PAMAP2<sup/);assert.match(html,/USC-HAD<sup/);
    assert.match(html,/pamap2_9c/);assert.match(html,/pamap2_12c/);
    assert.match(html,/usc_had_7c/);assert.match(html,/usc_had_12c/);
    assert.doesNotMatch(html,/Partial coverage|unranked|available-only/);
    assert.equal(app.get('reference-footnote').hidden,false);
    assert.equal(app.get('mode-controls').hidden,true);
    assert.equal(app.get('protocol-field').hidden,true);
  }
  assert.match(read('index.html'),/各占 25%/);
  assert.match(read('index.html'),/训练来源或标注预算/);
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
  const app=await boot('/leaderboard');app.get('sort-results').dispatch('click');
  const html=app.get('leaderboard-wrap').innerHTML;
  assert.ok(html.indexOf('data-method="onellm"')<html.indexOf('data-method="onehar"'));
  assert.match(html,/<tr class="reference-row" data-method="onellm"><td><span class="rank-number">19<\/span>/);
  assert.match(html,/<tr class="reference-row leading" data-method="onehar"><td><span class="rank-number">1<\/span>/);
});
test('overall Details separate the shared targets from other reported targets',async()=>{
  const app=await boot('/leaderboard');
  app.get('leaderboard-wrap').dispatch('click',{target:{closest:s=>s==='[data-summary]'?{dataset:{summary:'onehar'}}:null}});
  assert.match(app.get('detail-context').textContent,/4 datasets/);
  assert.match(app.get('detail-body').innerHTML,/Other reported datasets \(1\) · excluded/);
  assert.match(app.get('detail-body').innerHTML,/91\.90/);
  app.get('detail-body').dispatch('click',{target:{closest:()=>({dataset:{result:'transfer:onehar-joint:onehar:uci_har_6c'}})}});
  assert.match(app.get('detail-body').innerHTML,/Table 2 · PDF p\.16/);
  assert.match(app.get('detail-body').innerHTML,/https:\/\/dl\.acm\.org\/doi\/10\.1145\/3749509/);
});
test('each dataset view retains only source-paper methods, settings and the selected label set',async()=>{
  const app=await boot('/leaderboard','?board=imu&view=ranking&dataset=pamap2_9c');
  assert.match(app.get('dataset-select').innerHTML,/pamap2_9c/);assert.match(app.get('dataset-select').innerHTML,/pamap2_12c/);
  assert.equal((app.get('dataset-select').innerHTML.match(/<option /g)||[]).length,21);
  assert.match(app.get('dataset-heading').textContent,/PAMAP2 · 9 classes/);
  assert.match(app.get('leaderboard-wrap').innerHTML,/oneHAR/);
  assert.doesNotMatch(app.get('leaderboard-wrap').innerHTML,/MOSAIC|TinyHAR|RobustHAR/);
  const full=await boot('/leaderboard','?view=ranking&dataset=wisdm_18c&protocol=mosaic-full-shot');
  assert.match(full.get('result-summary').textContent,/8 models · 1 setting/);
  assert.match(full.get('leaderboard-wrap').innerHTML,/SensorLLM|GPT4TS/);
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
