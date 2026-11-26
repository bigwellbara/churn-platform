#!/usr/bin/env bash
# One-time Google Cloud setup for the deploy pipeline (.github/workflows/deploy.yml).
#
# Idempotent: re-running it creates only what is missing and re-applies settings.
# Needs: gcloud (logged in as a project Owner) and, optionally, the GitHub CLI `gh`
# (to store the results as repository variables).
#
#   PROJECT_ID=my-project bash infra/gcp/bootstrap.sh
#
# Optional: REGION (default africa-south1, Johannesburg), GITHUB_REPO (owner/name),
# BUDGET (e.g. 100ZAR or 5USD, in your billing account's currency) to add a budget alert.
#
# What it creates, and why (details: docs/DEPLOYMENT.md):
#   * Artifact Registry repo   private Docker registry for the API image, with a cleanup
#                              policy so old images don't pile up storage costs
#   * runtime service account  the identity the API runs as; it has no permissions,
#                              because the API needs none (least privilege)
#   * deployer service account what GitHub Actions acts as: may push images and deploy
#                              Cloud Run, nothing else
#   * Workload Identity Pool   lets GitHub Actions log in as the deployer with a
#                              short-lived OIDC token, only from this repo's main branch.
#                              No service-account key is ever created.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, e.g. PROJECT_ID=my-project bash $0}"
REGION="${REGION:-africa-south1}"
GITHUB_REPO="${GITHUB_REPO:-bigwellbara/churn-platform}"
REPOSITORY=churn
POOL=github
PROVIDER=github-oidc
DEPLOY_SA="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_SA="churn-api-runtime@${PROJECT_ID}.iam.gserviceaccount.com"

step() { printf '\n==> %s\n' "$*"; }
exists() { "$@" >/dev/null 2>&1; }

step "Project ${PROJECT_ID}"
gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
echo "project number ${PROJECT_NUMBER}"

step "Enabling APIs (first run takes a minute)"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com

step "Artifact Registry repository '${REPOSITORY}' in ${REGION}"
if ! exists gcloud artifacts repositories describe "$REPOSITORY" --location "$REGION"; then
  gcloud artifacts repositories create "$REPOSITORY" --location "$REGION" \
    --repository-format docker --description "churn-platform API images"
fi
# Keep the 10 newest images (rollback targets); delete anything else older than 30 days.
POLICY=$(mktemp)
cat >"$POLICY" <<'JSON'
[
  {"name": "keep-recent", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 10}},
  {"name": "delete-old", "action": {"type": "Delete"},
   "condition": {"tagState": "ANY", "olderThan": "30d"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies "$REPOSITORY" --location "$REGION" \
  --policy "$POLICY" --no-dry-run >/dev/null
rm -f "$POLICY"

step "Service accounts"
exists gcloud iam service-accounts describe "$RUNTIME_SA" ||
  gcloud iam service-accounts create churn-api-runtime \
    --display-name "churn-api (Cloud Run runtime, no permissions)"
exists gcloud iam service-accounts describe "$DEPLOY_SA" ||
  gcloud iam service-accounts create github-deployer \
    --display-name "GitHub Actions deployer for ${GITHUB_REPO}"

step "Deployer permissions"
# Deploy services and make them public (setting the invoker policy needs run.admin).
gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition=None \
  --member "serviceAccount:${DEPLOY_SA}" --role roles/run.admin >/dev/null
# Push images to this one repository only.
gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" --location "$REGION" \
  --member "serviceAccount:${DEPLOY_SA}" --role roles/artifactregistry.writer >/dev/null
# Start revisions that run as the runtime account (and only that account).
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member "serviceAccount:${DEPLOY_SA}" --role roles/iam.serviceAccountUser >/dev/null

step "Workload Identity Federation (GitHub OIDC -> deployer)"
if ! exists gcloud iam workload-identity-pools describe "$POOL" --location global; then
  gcloud iam workload-identity-pools create "$POOL" --location global \
    --display-name "GitHub Actions"
fi
# Tokens are accepted only from this repository's main branch.
PROVIDER_ARGS=(
  --location global
  --workload-identity-pool "$POOL"
  --issuer-uri "https://token.actions.githubusercontent.com"
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref"
  --attribute-condition "assertion.repository=='${GITHUB_REPO}' && assertion.ref=='refs/heads/main'"
)
if exists gcloud iam workload-identity-pools providers describe "$PROVIDER" \
  --location global --workload-identity-pool "$POOL"; then
  gcloud iam workload-identity-pools providers update-oidc "$PROVIDER" "${PROVIDER_ARGS[@]}"
else
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" "${PROVIDER_ARGS[@]}" \
    --display-name "GitHub ${GITHUB_REPO}"
fi
POOL_ID="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  >/dev/null
WIF_PROVIDER="${POOL_ID}/providers/${PROVIDER}"

if [ -n "${BUDGET:-}" ]; then
  step "Budget alert (${BUDGET}/month: emails at 50%, 90%, 100%)"
  gcloud services enable billingbudgets.googleapis.com
  BILLING=$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingAccountName)')
  BILLING="${BILLING#billingAccounts/}"
  if gcloud billing budgets list --billing-account "$BILLING" --format='value(displayName)' |
    grep -qx churn-platform; then
    echo "budget 'churn-platform' already exists"
  else
    gcloud billing budgets create --billing-account "$BILLING" --display-name churn-platform \
      --budget-amount "$BUDGET" --filter-projects "projects/${PROJECT_ID}" \
      --threshold-rule percent=0.5 --threshold-rule percent=0.9 --threshold-rule percent=1.0
  fi
fi

step "GitHub repository variables"
declare -A VARS=(
  [GCP_PROJECT_ID]="$PROJECT_ID"
  [GCP_REGION]="$REGION"
  [GCP_WIF_PROVIDER]="$WIF_PROVIDER"
  [GCP_DEPLOY_SA]="$DEPLOY_SA"
  [GCP_RUNTIME_SA]="$RUNTIME_SA"
)
# None of these are secrets: they say *who* may deploy, they don't grant it.
for name in "${!VARS[@]}"; do
  if command -v gh >/dev/null; then
    gh variable set "$name" --repo "$GITHUB_REPO" --body "${VARS[$name]}"
  else
    echo "$name=${VARS[$name]}"
  fi
done
command -v gh >/dev/null || echo "(gh not found: add the variables above under Settings > Secrets and variables > Actions > Variables)"

step "Done. Push to main (or run the Deploy workflow by hand) to deploy."
