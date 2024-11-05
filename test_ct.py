import pytest
import itertools
import json
from time import sleep, time
from tapipy.tapis import Tapis

base_url = 'https://icicleai.develop.tapis.io'
models = ['41d3ed40-b836-4a62-b3fb-67cee79f33d9-model'] #, '4108ed9d-968e-4cfe-9f18-0324e5399a97-model', '665e7c60-7244-470d-8e33-a232d5f2a390-model']
devices = ['x86']#, 'Jetson', 'compute_cascadelake', 'gpu_k80', 'gpu_p100']
sites = ['TACC']#, 'CHI@TACC']
DEBUG = False
DEBUG = './test-job-logs'

@pytest.fixture(scope='session', autouse=True)
def experiment_logs(tmp_path_factory):
    if DEBUG:
        yield DEBUG
    else:
        yield tmp_path_factory.mktemp("experiments")

@pytest.fixture(scope="session", autouse=True)
def tapis_client():
    with open('credentials.json', 'r') as f:
        cred = json.load(f)
        username = cred['username']
        password = cred['password']
    t = Tapis(base_url=base_url, username=username, password=password)
    t.get_tokens()
    yield t

    print('\n\nshutting down the token\n\n')

@pytest.fixture(params=[(model, device, site) for model in models for device in devices for site in sites])
def ct_job_completed(request, tapis_client, experiment_logs):
    #print(request)
    if DEBUG:
        yield True
    else:
        model = request.param[0]
        device = request.param[1]
        site = request.param[2]
        #print(f'running {model} on {device} at {site}')
        submission = generate_submission(model, device, site)
        # submit job
        jobinfo = tapis_client.jobs.submitJob(name=submission['name'],
                                              description=submission['description'],
                                              appId=submission['appId'],
                                              appVersion=submission['appVersion'],
                                              parameterSet=submission['parameterSet'])
        # get job id
        tapisjobid = jobinfo.get('uuid')
        # check done
        completed(tapisjobid, tapis_client)
        print('job has completed, now running tests')
        with open(f'{experiment_logs}/{model}x{device}x{site}.out', 'w') as f:
            f.write(tapisjobid)
        yield True

def job_running(jobid, tapis_client):
    jobinfo = tapis_client.jobs.getJob(jobUuid=jobid)
    status = jobinfo.get('status')
    #print(f'status is {status}')
    #if status == 'RUNNING' or status == 'PENDING' or status == 'STAGING_JOB':
    if status in ['PROCESSING_INPUTS', 'STAGING_JOB', 'PENDING', 'RUNNING', 'ARCHIVING']:
        return True
    else:
        return False

def completed(jobid, tapis_client):
    completed = False
    interval_time = 10
    max_wait = 3600
    start_time = time()
    while True:
        if not job_running(jobid, tapis_client):
            completed = True
            break

        elapsed_time = time() - start_time
        if elapsed_time >= max_wait:
            break
        sleep(interval_time)
    return completed

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


@pytest.mark.parametrize("model, device, site", list(itertools.product(models, devices, sites)))
class TestCameraTraps:
    def get_job_id(self, model, device, site, experiment_logs):
        with open(f'{experiment_logs}/{model}x{device}x{site}.out', 'r') as f:
            jobid = f.readline()
        return jobid

    def test_completes(self, model, device, site, tapis_client, ct_job_completed, experiment_logs):
        jobid = self.get_job_id(model, device, site, experiment_logs)
        print(f'experiment logs: {experiment_logs}')
        #print(f'Testing {model} on a {device} at {site} jobid={jobid}')
        print('Waiting for job to complete')
        if ct_job_completed:
            pass
        assert tapis_client.jobs.getJob(jobUuid=jobid).get('status') == 'FINISHED'

    def test_image_files_exist(self, model, device, site, tapis_client, experiment_logs):
        jobid = self.get_job_id(model, device, site, experiment_logs)
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        files = tapis_client.files.listFiles(systemId='icicledev-test', path=jobdir+'/ct_run/images_output_dir')
        num_scores = [file for file in files if '.score' not in file.name]
        assert len(num_scores) == 6

    def test_score_files_exist(self, model, device, site, tapis_client, experiment_logs):
        jobid = self.get_job_id(model, device, site, experiment_logs)
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        files = tapis_client.files.listFiles(systemId='icicledev-test', path=jobdir+'/ct_run/images_output_dir')
        num_scores = [file for file in files if '.score' in file.name]
        assert len(num_scores) == 6

    def test_power_data(self, model, device, site, tapis_client, experiment_logs):
        jobid = self.get_job_id(model, device, site, experiment_logs)
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        power_summary_str = tapis_client.files.getContents(systemId='icicledev-test', path=jobdir+'/ct_run/power_output_dir/power_summary_report.json')
        power_summary = json.loads(power_summary_str.decode('utf-8'))