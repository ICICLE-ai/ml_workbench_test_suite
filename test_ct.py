import pytest
import json
import yaml
from time import sleep, time
from tapipy.tapis import Tapis
import io
import os
import paramiko

base_url = 'https://icicleai.tapis.io'
#DEBUG = './test-job-logs'
DEBUG = False

def read_inputs(input_yml):
    with open(input_yml, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['models'], cfg['device_map'], cfg['datasets'], cfg['custom_app_vars']

models, device_map, datasets, custom_app_vars = read_inputs('input.yml')

def get_expected_images(model, dataset='15-image', parameter='images'):
    with open('expected_values.json', 'r') as f:
        expected_values = json.load(f)
    return expected_values['experiments'][model][dataset][parameter]

def get_all_experiment_ids():
    all_experiment_ids = []
    for model in models:
        for dataset in datasets:
            for site, devices in device_map.items():
                for device in devices:
                    all_experiment_ids.append(f'{site}_{device}_{model}_{dataset}')
    return all_experiment_ids

def get_all_experiments():
    all_experiments = []
    for model in models:
        for dataset in datasets:
            for site, devices in device_map.items():
                for device in devices:
                    all_experiments.append((model, device, site, dataset))
    return all_experiments

@pytest.fixture(scope='session', autouse=True)
def experiment_logs(tmp_path_factory):
    if DEBUG:
        log_dir = DEBUG
    else:
        log_dir = tmp_path_factory.mktemp("experiments")
    yield log_dir

@pytest.fixture(scope="session", autouse=True)
def tapis_client():
    if os.path.exists('credentials.json'):
        with open('credentials.json', 'r') as f:
            cred = json.load(f)
            username = cred['username']
            password = cred['password']
    elif 'TAPIS_USER' in os.environ and 'TAPIS_PASSWORD' in os.environ:
        username = os.environ['TAPIS_USER']
        password = os.environ['TAPIS_PASSWORD']
    else:
        raise Exception('Tapis credentials not found')
    t = Tapis(base_url=base_url, username=username, password=password)
    t.get_tokens()
    yield t


def update_image():
    username = os.environ['HOST_USER']
    ip_address = os.environ['HOST_IP_ADDR']
    pkey = paramiko.RSAKey.from_private_key(io.StringIO(os.environ['HOST_PKEY']))
    client = paramiko.SSHClient()
    policy = paramiko.AutoAddPolicy()
    client.set_missing_host_key_policy(policy)
    client.connect(ip_address, username=username, pkey=pkey)
    _, stdout, stderr = client.exec_command('docker pull tapis/ctcontroller', get_pty=True)
    stderr = stderr.read().decode("utf-8").strip()
    if stderr:
        stdout = stdout.read().decode("utf-8").strip()
        print(f'Updating docker image:\n{stdout}\n{stderr}')
    client.close()

@pytest.fixture(params=get_all_experiments(), ids=get_all_experiment_ids())
def job_info(request, tapis_client, experiment_logs):
    model = request.param[0]
    device = request.param[1]
    site = request.param[2]
    dataset=request.param[3]
    # generate job submission
    submission = generate_submission(model, device, site, dataset)
    if os.path.exists(f'{experiment_logs}/{model}x{device}x{site}x{dataset}.out'):
        # if output file exists, get jobid from there and do not resubmit
        with open(f'{experiment_logs}/{model}x{device}x{site}x{dataset}.out', 'r') as f:
            tapisjobid = f.readline()
    else:
        # submit job
        update_image()
        jobinfo = tapis_client.jobs.submitJob(name=submission['name'],
                                              description=submission['description'],
                                              appId=submission['appId'],
                                              appVersion=submission['appVersion'],
                                              parameterSet=submission['parameterSet'])
        # get job id
        tapisjobid = jobinfo.get('uuid')
        # Write job info to log files
        with open(f'{experiment_logs}/{model}x{device}x{site}x{dataset}.json', 'w') as f:
            json.dump(submission, f)
        with open(f'{experiment_logs}/{model}x{device}x{site}x{dataset}.out', 'w') as f:
            f.write(tapisjobid)
    # poll until job has completed
    completed(tapisjobid, tapis_client)
    if not validate_provisioning(tapisjobid, tapis_client):
        raise pytest.skip(f'Resource {device} at {site} is not currently available')
    tapis_client.get_tokens()
    yield tapisjobid, model, device, site, dataset

def validate_provisioning(jobid, client):
    if client.jobs.getJob(jobUuid=jobid).get('status') == 'FAILED':
        jobdir = client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        log_file = client.files.getContents(systemId='icicledev-test', path=jobdir+'/run.log').decode('utf-8')
        if 'Try rerunning later' in log_file:
            return False
        else:
            return True
    else:
        return True

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

def generate_submission(model, device, site, dataset):
    d = {}
    d['name'] = f'testsuite_{site}_{device}_{dataset}_{model}'[:64]
    d['appId'] = 'cameratraps-test'
    d['appVersion'] = '0.1'
    d['description'] = f'Invoke ctcontroller to run camera-traps on {site} {device}'
    envVariables = []
    advanced_app_vars = {} 
    with open('expected_values.json', 'r') as f:
        expected_values = json.load(f)
    if expected_values['datasets'][dataset]['images'] != '':
        image_url = expected_values['datasets'][dataset]['images']
        advanced_app_vars['use_bundled_example_images'] = 'false'
        envVariables.append({'key': 'CT_CONTROLLER_INPUT', 'value': image_url})
    if expected_values['datasets'][dataset]['ground_truth'] != '':
        ground_truth_url = expected_values['datasets'][dataset]['ground_truth']
        advanced_app_vars['use_custom_ground_truth_file_url'] = 'true'
        advanced_app_vars['custom_ground_truth_file_url'] = expected_values['datasets'][dataset]['ground_truth']
    if custom_app_vars:
        advanced_app_vars.update(custom_app_vars)
    envVariables.append({'key': 'CT_CONTROLLER_TARGET_SITE', 'value': site})
    envVariables.append({'key': 'CT_CONTROLLER_NODE_TYPE', 'value': device})
    envVariables.append({'key': 'CT_CONTROLLER_GPU', 'value': enable_gpu(device)})
    envVariables.append({'key': 'CT_CONTROLLER_CONFIG_PATH', 'value': '/config.yml'})
    envVariables.append({'key': 'CT_CONTROLLER_MODEL', 'value': model})
    envVariables.append({'key': 'CT_CONTROLLER_ADVANCED_APP_VARS', 'value': json.dumps(advanced_app_vars)})
    if 'CT_VERSION' in os.environ:
        ct_version = os.environ['CT_VERSION']
    else:
        ct_version = 'latest'
    envVariables.append({'key': 'CT_CONTROLLER_CT_VERSION', 'value': ct_version})
    d['parameterSet'] = {'envVariables': envVariables}
    d['archiveFilter'] = {'includeLaunchFiles': False}
    return d


class TestCameraTraps:
    def test_completes(self, tapis_client, job_info):
        jobid, model, device, site, dataset = job_info
        # on job failure, get the tail of the job log
        if tapis_client.jobs.getJob(jobUuid=jobid).get('status') == 'FAILED':
            jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
            log_file = tapis_client.files.getContents(systemId='icicledev-test', path=jobdir+'/run.log')
            print('\n'.join(log_file.decode('utf-8').split('\n')[-20:]))
        assert tapis_client.jobs.getJob(jobUuid=jobid).get('status') == 'FINISHED'

    def test_image_files_exist(self, tapis_client, job_info):
        jobid, model, device, site, dataset = job_info
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        files = tapis_client.files.listFiles(systemId='icicledev-test', path=jobdir+'/ct_run/images_output_dir')
        num_images = [file for file in files if '.jpeg' not in file.name]
        assert len(num_images) == get_expected_images(model, dataset=dataset, parameter='images')

    def test_score_files_exist(self, tapis_client, job_info):
        jobid, model, device, site, dataset = job_info
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        files = tapis_client.files.listFiles(systemId='icicledev-test', path=jobdir+'/ct_run/images_output_dir')
        num_scores = [file for file in files if '.score' in file.name]
        assert len(num_scores) == get_expected_images(model, dataset=dataset, parameter='scores')

    def test_power_data(self, tapis_client, job_info):
        jobid, model, device, site, dataset = job_info
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        power_summary_str = tapis_client.files.getContents(systemId='icicledev-test', path=jobdir+'/ct_run/power_output_dir/power_summary_report.json')
        power_summary = json.loads(power_summary_str.decode('utf-8'))
        # allow for the image generating plugin to complete too quickly to capture power usage
        cpu_plugins = [plugin['cpu_power_consumption']>0 for plugin in power_summary['plugin power summary report']]
        if sum(cpu_plugins) < len(cpu_plugins) and sum(cpu_plugins) > 0 and \
            device != 'Jetson' and dataset == '15-image':
            pytest.xfail(reason=f'{device} is too fast to capture power usage')
        assert sum(cpu_plugins) == len(cpu_plugins)
        if enable_gpu(device) == 'true':
            gpu_plugins = [plugin['gpu_power_consumption']>0 for plugin in power_summary['plugin power summary report']]
            if device != 'Jetson' and dataset == '15-image':
                if sum(gpu_plugins) < len(gpu_plugins):
                    pytest.xfail(reason=f'{device} is too fast to capture GPU power usage')
            assert sum(gpu_plugins) >= len(gpu_plugins) - 1
        else:
            assert all([plugin['gpu_power_consumption']==0 for plugin in power_summary['plugin power summary report']])

    def test_ckn_events(self, tapis_client, job_info):
        jobid, model, device, site, dataset = job_info
        jobdir = tapis_client.jobs.getJob(jobUuid=jobid).get('archiveSystemDir')
        ckn_events = tapis_client.files.getContents(systemId='icicledev-test', path=jobdir+'/ct_run/oracle_output_dir/ckn.log')
        events = [line for line in ckn_events.decode('utf-8').split('\n') if 'New oracle event' in line]
        assert len(events) == get_expected_images(model, dataset=dataset, parameter='ckn_events')
