import pytest
import itertools
import json
from time import sleep, time
from tapipy.tapis import Tapis
import os

base_url = 'https://icicleai.develop.tapis.io'
models = ['41d3ed40-b836-4a62-b3fb-67cee79f33d9-model', '4108ed9d-968e-4cfe-9f18-0324e5399a97-model', '665e7c60-7244-470d-8e33-a232d5f2a390-model']
devices = ['x86', 'Jetson', 'compute_cascadelake', 'gpu_k80', 'gpu_p100']
sites = ['TACC', 'CHI@TACC']
#DEBUG = './test-job-logs'
DEBUG = False

@pytest.fixture(scope='session', autouse=True)
def experiment_logs(tmp_path_factory):
    if DEBUG:
        log_dir = DEBUG
    else:
        log_dir = tmp_path_factory.mktemp("experiments")
    yield log_dir

@pytest.fixture(scope="session", autouse=True)
def tapis_client():
    with open('credentials.json', 'r') as f:
        cred = json.load(f)
        username = cred['username']
        password = cred['password']
    t = Tapis(base_url=base_url, username=username, password=password)
    t.get_tokens()
    yield t

@pytest.fixture(params=[(model, device, site) for model in models for device in devices for site in sites])
def job_info(request, tapis_client, experiment_logs):
    model = request.param[0]
    device = request.param[1]
    site = request.param[2]
    # generate job submission
    submission = generate_submission(model, device, site)
    if os.path.exists(f'{experiment_logs}/{model}x{device}x{site}.out'):
        # if output file exists, get jobid from there and do not resubmit
        with open(f'{experiment_logs}/{model}x{device}x{site}.out', 'r') as f:
            tapisjobid = f.readline()
    else:
        # submit job
        jobinfo = tapis_client.jobs.submitJob(name=submission['name'],
                                              description=submission['description'],
                                              appId=submission['appId'],
                                              appVersion=submission['appVersion'],
                                              parameterSet=submission['parameterSet'])
        # get job id
        tapisjobid = jobinfo.get('uuid')
        # Write job info to log files
        with open(f'{experiment_logs}/{model}x{device}x{site}.json', 'w') as f:
            json.dump(submission, f)
        with open(f'{experiment_logs}/{model}x{device}x{site}.out', 'w') as f:
            f.write(tapisjobid)
    # poll until job has completed
    completed(tapisjobid, tapis_client)
    yield tapisjobid, model, device, site

def job_running(jobid, tapis_client):
    jobinfo = tapis_client.jobs.getJob(jobUuid=jobid)
    status = jobinfo.get('status')
    if status in ['PROCESSING_INPUTS', 'STAGING_JOB', 'PENDING', 'RUNNING', 'ARCHIVING']:
        return True
    else:
        return False

def completed(jobid, tapis_client):
    job_completed = False
    interval_time = 10
    max_wait = 3600
    start_time = time()
    while True:
        if not job_running(jobid, tapis_client):
            job_completed = True
            break

        elapsed_time = time() - start_time
        if elapsed_time >= max_wait:
            break
        sleep(interval_time)
    return job_completed

def enable_gpu(device: str) -> str:
    if 'gpu' in device or device == 'Jetson':
        return 'true'
    else:
        return 'false' 

def generate_submission(model, device, site):
    d = {}
    d['name'] = f'testsuite_{site}_{device}_{model}'
    d['appId'] = 'cameratraps-test'
    d['appVersion'] = '0.1'
    d['description'] = f'Invoke ctcontroller to run camera-traps on {site} {device}'
    envVariables = []
    envVariables.append({'key': 'CT_CONTROLLER_TARGET_SITE', 'value': site})
    envVariables.append({'key': 'CT_CONTROLLER_NODE_TYPE', 'value': device})
    envVariables.append({'key': 'CT_CONTROLLER_GPU', 'value': enable_gpu(device)})
    envVariables.append({'key': 'CT_CONTROLLER_CONFIG_PATH', 'value': '/config.yml'})
    d['parameterSet'] = {'envVariables': envVariables}
    d['archiveFilter'] = {'includeLaunchFiles': False}
    return d


class TestCameraTraps:
    def test_completes(self, tapis_client, job_info):
        jobid, model, device, site = job_info
        assert tapis_client.jobs.getJob(jobUuid=jobid).get('status') == 'FINISHED'

    def test_image_files_exist(self, tapis_client, job_info):
        jobid, model, device, site = job_info
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        files = tapis_client.files.listFiles(systemId='icicledev-test', path=jobdir+'/ct_run/images_output_dir')
        num_scores = [file for file in files if '.score' not in file.name]
        assert len(num_scores) == 6

    def test_score_files_exist(self, tapis_client, job_info):
        jobid, model, device, site = job_info
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        files = tapis_client.files.listFiles(systemId='icicledev-test', path=jobdir+'/ct_run/images_output_dir')
        num_scores = [file for file in files if '.score' in file.name]
        assert len(num_scores) == 6

    def test_power_data(self, tapis_client, job_info):
        jobid, model, device, site = job_info
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        power_summary_str = tapis_client.files.getContents(systemId='icicledev-test', path=jobdir+'/ct_run/power_output_dir/power_summary_report.json')
        power_summary = json.loads(power_summary_str.decode('utf-8'))
        assert all([plugin['cpu_power_consumption']>0 for plugin in power_summary['plugin power summary report']])
        if enable_gpu(device) == 'true':
            assert all([plugin['gpu_power_consumption']>0 for plugin in power_summary['plugin power summary report']])
        else:
            assert all([plugin['gpu_power_consumption']==0 for plugin in power_summary['plugin power summary report']])

    def test_ckn_events(self, tapis_client, job_info):
        jobid, model, device, site = job_info
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        ckn_events = tapis_client.files.getContents(systemId='icicledev-test', path=jobdir+'/ct_run/oracle_output_dir/ckn.log')
        events = [line for line in ckn_events.decode('utf-8').split('\n') if 'New oracle event' in line]
        assert len(events) == 12