from github import Github

# GitHub repository information
repo_owner = "dancosta154"
github_token = "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"


def get_github_instance():
    # Create a GitHub instance using your personal access token
    g = Github(github_token)
    return g


def get_user_repos(github_instance):
    # Get the user or organization
    user = github_instance.get_user(repo_owner)

    # Get all repositories for the user or organization
    repos = user.get_repos()
    return repos
