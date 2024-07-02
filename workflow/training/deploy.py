import datetime

from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings
from github import Auth as GithubAuth
from github import Github

from workflow.training.utils import download_model, push_to_hub

logger = get_task_logger(__name__)


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def deploy_model(self, request_data):
    finetuned_model_id = request_data["finetuned_model"]
    deployment_model_id = request_data["deployment_model"]
    gh_workflow = request_data["gh_workflow"]
    service_names = request_data["service_names"]
    try:
        model_path = download_model(repo_id=finetuned_model_id)
        logger.debug(f"Downloaded model from Hugging Face Hub: {finetuned_model_id}")
        push_to_hub(
            folder_path=model_path, repo_id=deployment_model_id, repo_type="model"
        )
        logger.debug(f"Pushed model to Hugging Face Hub: {deployment_model_id}")

        run_github_workflow(
            workflow_name=gh_workflow, inputs={"service_names": service_names}
        )

        # TODO: Generalise the deployment workflow
        run_github_workflow(
            workflow_name="deploy-service.yaml",
            repo="BharatSahAIyak/docker-bhasai",
            branch="dev",
            inputs={
                "profiles": "application database",
                "environment": "dev",
                "services": "ai-tools",
            },
        )
    except Exception as e:
        logger.error(f"Failed to deploy model: \n{e}")


def run_github_workflow(
    workflow_name: str,
    repo: str = settings.AI_TOOLS_REPO,
    branch: str = settings.AI_TOOLS_REPO_BRANCH,
    inputs: dict = {},
):
    try:
        logger.info("Dispatching the workflow: {}.".format(workflow_name))
        g = Github(auth=GithubAuth.Token(settings.GITHUB_PAT))
        repo = g.get_repo(repo)
        ref = repo.get_branch(branch)
        workflow = repo.get_workflow(workflow_name)
        now = datetime.datetime.now()

        is_dispatched = workflow.create_dispatch(ref=ref, inputs=inputs)
        if not is_dispatched:
            raise Exception(
                "Failed to dispatch the workflow: {}.".format(workflow_name)
            )

        logger.info("Waiting for the workflow to start.")
        while (
            repo.get_workflow_runs(
                status="in_progress",
                created=">="
                + (now - datetime.timedelta(seconds=5)).strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00"
                ),
            ).totalCount
            == 0
        ):
            pass

        run = list(
            repo.get_workflow_runs(
                status="in_progress",
                created=">="
                + (now - datetime.timedelta(seconds=5)).strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00"
                ),
            )
        )[0]

        logger.info(f"Waiting for the workflow {run.id} to complete.")
        while True:
            run = repo.get_workflow_run(run.id)
            if (
                run.status == "completed"
                or run.status == "failure"
                or run.status == "cancelled"
            ):
                break

        logger.info(f"{run.id} : {run.status}")
        logger.info(run.conclusion)

        g.close()
    except Exception as e:
        logger.error(f"Failed to run the workflow: \n{e}")
