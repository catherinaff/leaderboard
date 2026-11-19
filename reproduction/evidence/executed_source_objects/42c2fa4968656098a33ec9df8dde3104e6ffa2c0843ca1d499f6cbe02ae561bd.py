"""Non-benchmark contract checks, including disabled paid network paths."""
from pathlib import Path
import json
import socket
import os
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from api_adapters import GuardedChat,PaidCallsDisabled,parse_answer,atomic_json
from onellm_adapter import flash_sdpa,imu_window
from sensorllm_adapter import original_qa_functions

p=Path(__file__).resolve().parent
checks=[]
for configured,execute in [(False,False),(False,True),(True,False)]:
 with patch.object(socket,'create_connection',side_effect=AssertionError('No paid network allowed')):
  client=GuardedChat(dict(allow_paid_calls=configured,model=None),p/'contract_fixtures_20261113/api',execute=execute)
  try:client.generate([dict(type='text',text='Offline guard fixture')])
  except PaidCallsDisabled:pass
  else:raise AssertionError('A disabled route reached the client')
checks.append('All three disabled combinations stop before SDK/client/network construction')
# Exercise real SDK serialization, response parsing, caching and the persistent
# limit against a local HTTP mock. No network, provider, credential or charge.
import httpx
import openai
sdk=openai.OpenAI;requests=[]
def respond(request):
    body=json.loads(request.content);requests.append(body)
    assert body['messages'][0]['content'][0]['type']=='text'
    return httpx.Response(200,json=dict(id='offline-fixture',object='chat.completion',created=0,model='offline-fixture',
        choices=[dict(index=0,message=dict(role='assistant',content='<answer>Walking</answer>'),finish_reason='stop')],
        usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15)))
def factory(**kwargs):return sdk(**kwargs,http_client=httpx.Client(transport=httpx.MockTransport(respond)))
config=dict(allow_paid_calls=True,model='offline-fixture',api_key_env='HAR_OFFLINE_FIXTURE_KEY_20261113',
            max_calls=1,budget_usd=1,max_request_cost_usd=1,request_cost_bound_verified=True)
with tempfile.TemporaryDirectory(prefix='har-sdk-offline-') as cache:
 with patch.object(openai,'OpenAI',factory),patch.dict(os.environ,{'HAR_OFFLINE_FIXTURE_KEY_20261113':'offline-fixture-only'}),patch.object(socket.socket,'connect',side_effect=AssertionError('No network in offline SDK test')):
  client=GuardedChat(config,cache,execute=True)
  payload=[dict(type='text',text='Offline serialization fixture')]
  assert client.generate(payload)=='<answer>Walking</answer>'
  assert client.generate(payload)=='<answer>Walking</answer>' and len(requests)==1
  try:client.generate([dict(type='text',text='Second uncached fixture')])
  except PaidCallsDisabled:pass
  else:raise AssertionError('Persistent limit must stop a second request')
  assert len(requests)==1
checks.append('Real SDK with network blocked: payload, response, cache hit and persistent request limit verified locally; zero paid calls')
assert parse_answer('<answer>Walking Upstairs</answer>',['Walking','Walking Upstairs'])==1
try:parse_answer('<answer>Walking or Walking Upstairs</answer>',['Walking','Walking Upstairs'])
except ValueError:pass
else:raise AssertionError('Ambiguous prediction must not become a score')
checks.append('Exact label parsing rejects ambiguous overlapping activity names')
torch.manual_seed(42)
for nq,nk,causal in [(7,7,True),(1,7,True),(3,7,True),(4,4,False)]:
 q=torch.randn(2,nq,4,8);k=torch.randn(2,nk,4,8);v=torch.randn(2,nk,4,8)
 scores=torch.einsum('bqhd,bkhd->bhqk',q,k)/8**.5
 if causal:
  allowed=torch.arange(nk)[None,:]<=torch.arange(nq)[:,None]+nk-nq
  scores=scores.masked_fill(~allowed,float('-inf'))
 expected=torch.einsum('bhqk,bkhd->bqhd',scores.softmax(-1),v)
 actual=flash_sdpa(q,k,v,causal=causal)
 torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
checks.append('OneLLM SDPA matches explicit attention, including unequal cached query/key lengths')
for dataset,channels,length in [('uci_har',9,128),('pamap2',27,200),('usc_had',6,200),('realworld_har',42,100)]:
 data=np.tile(np.arange(channels,dtype='float32'),(length,1))
 window=imu_window(data,dataset);assert window.shape==(6,400) and np.isfinite(window).all()
 if dataset=='uci_har':assert np.allclose(window[:3,80:120].mean(1),[6,7,8],atol=.01)
checks.append('OneLLM physical channel selection/rate/window conversion, UCI total_acc order verified')
qa=original_qa_functions();signal=np.sin(np.arange(100)/8)
frame=qa['merge_adjacent_rows'](qa['analyze_trend'](signal,50));pair=qa['select_random_pair']()
trend=qa['dscb_simple_trend'](frame,'sensor',pair);summary=qa['QA_summary'](signal,frame,'sensor',pair)
assert trend['Q'] and trend['A'] and summary['Q'] and summary['A']
checks.append('SensorLLM original notebook generates trend/summary QA locally without an API')
atomic_json(p/'audits/adapter_contract_checks_20261113.json',dict(status='passed',paid_api_calls=0,benchmark_result=False,checks=checks))
print(json.dumps(dict(status='passed',checks=checks),indent=2),flush=True)
