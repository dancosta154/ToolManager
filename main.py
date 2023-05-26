import sys
from modeling import preprocess_text, create_word2vec_model, calculate_cosine_similarity
from creds import get_github_instance, get_user_repos
import warnings


# Suppress NLTK data download messages
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")

# User input string
input_string = sys.argv[1]

# Get the GitHub instance and user repositories
github_instance = get_github_instance()
repos = get_user_repos(github_instance)

# Create a list to store all stemmed tokens
all_tokens = []

# Iterate over repositories and collect tokens
for repo in repos:
    # Fetch the README file for the repository
    readme_file = repo.get_readme()

    # Download the README file content
    readme_content = readme_file.decoded_content.decode("utf-8")

    # Preprocess the readme content and collect tokens
    tokens = preprocess_text(readme_content)
    all_tokens.extend(tokens)

# Create Word2Vec model for the stemmed tokens
model = create_word2vec_model(all_tokens)

# Calculate cosine similarity and common words for each repository's README
results = []
for repo in repos:
    # Fetch the README file for the repository
    readme_file = repo.get_readme()

    # Download the README file content
    readme_content = readme_file.decoded_content.decode("utf-8")

    # Calculate the cosine similarity
    cosine_similarity_score, common_words = calculate_cosine_similarity(
        model, input_string, readme_content
    )

    results.append((cosine_similarity_score, repo.name, readme_content, common_words))

similarity_matrix = {
    0.9: "Very High Similarity",
    0.8: "High Similarity",
    0.7: "High Similarity",
    0.6: "Moderate Similarity",
    0.5: "Moderate Similarity",
    0.4: "Moderate Similarity",
    0.3: "Low Similarity",
    0.2: "Low Similarity",
    0.1: "Low Similarity",
    0.0: "No Similarity",
}

# Sort the results based on cosine similarity score in descending order
results.sort(reverse=True)

print("<<<<<<<< Cosine Similarity >>>>>>>>")
print("\n")
for score, repo_name, readme_content, common_words in results:
    print("Repository:", repo_name)
    print("Cosine similarity score:", score)
    for threshold, interpretation in similarity_matrix.items():
        if score >= threshold:
            print("Interpretation:", interpretation)
            break
    # Print common words
    if len(common_words) > 0:
        print("\n")
        print("Common Words:")
        print(", ".join(common_words))
        print("\n")
    else:
        print("\n")
    print("<<<<<<<<>>>>>>>>")
