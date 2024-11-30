# Rule-Based Chatbot

A rule-based chatbot implementation using Natural Language Processing (NLP) techniques and machine learning for intelligent conversation handling.

## Features

- Advanced text preprocessing and tokenization
- Context-aware responses using conversation history
- Multiple response strategies:
  - Context-based matching
  - Similarity-based matching
  - Keyword-based matching
- Error handling and logging

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .botvenv
   source .botvenv/bin/activate  # On Windows, use `.botvenv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the chatbot, execute one of the following scripts:

- For the basic version:

  ```bash
  python rule_based/rule_based_v2.py
  ```

- For the enhanced version with additional features:
  ```bash
  python rule_based/rule_based_v3.py
  ```

## Project Structure

- `rule_based/`: Contains the main chatbot implementations.
- `requirements.txt`: Lists all the Python dependencies.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `README.md`: Project documentation.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
