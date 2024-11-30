import pickle
import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class Chatbot:
    def __init__(self) -> None:
        """
        Initialise the RuleBasedChatbot by downloading necessary NLTK data files,
        initialising preprocessing tools, and loading data to build response dictionaries.
        """
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.stop_words: set[str] = set(stopwords.words('english'))
        
        self.load_data()
        self.build_response_dictionaries()

    def load_data(self) -> None:
        """
        Load the conversations dataset and create a message lookup dictionary.
        If the message lookup dictionary does not exist, it is created and saved.
        """
        with open('./processed_data/processed_conversations.pkl', 'rb') as f:
            self.conversations_dict: dict[str, list[dict]] = pickle.load(f)

        try:
            with open('./processed_data/message_lookup.pkl', 'rb') as f:
                self.message_lookup: dict[str, dict] = pickle.load(f)

        except FileNotFoundError:
            self.message_lookup = {}
            for conv_id, messages in self.conversations_dict.items():
                for message in messages:
                    self.message_lookup[message['id']] = message
                    
            with open('./processed_data/message_lookup.pkl', 'wb') as f:
                pickle.dump(self.message_lookup, f)

    def preprocess(self, text: str) -> str:
        """
        Preprocess the input text by converting it to lowercase,
        tokenising (including punctuation removal), and lemmatising the words.

        Args:
            text: The input text to preprocess.

        Returns:
            The preprocessed text.
        """
        text = text.lower()
        tokens = wordpunct_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token.isalnum()]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def get_keywords(self, text: str) -> list[str]:
        """
        Extract keywords from the input text by removing stopwords.

        Args:
            text: The input text to extract keywords from.

        Returns:
            A list of keywords.
        """
        tokens = word_tokenize(text)
        keywords = [word for word in tokens if word not in self.stop_words]
        return keywords

    def build_response_dictionaries(self) -> None:
        """
        Build response and keyword dictionaries from conversation pairs.
        """
        conversation_pairs: list[tuple[str, str]] = []
        for conv_id, messages in self.conversations_dict.items():
            for message in messages:
                if 'text' in message and message.get('reply_to'):
                    reply_to_id = message['reply_to']
                    if reply_to_id in self.message_lookup:
                        user_message = self.message_lookup[reply_to_id]['text']
                        bot_response = message['text']
                        conversation_pairs.append((user_message, bot_response))

        self.response_dict: dict[str, str] = {}
        self.keyword_dict: dict[str, list[str]] = {}
        
        for user_input, bot_response in conversation_pairs:
            preprocessed_input = self.preprocess(user_input)
            self.response_dict[preprocessed_input] = bot_response
            
            keywords = self.get_keywords(preprocessed_input)
            for keyword in keywords:
                if keyword in self.keyword_dict:
                    self.keyword_dict[keyword].append(bot_response)
                else:
                    self.keyword_dict[keyword] = [bot_response]

    def get_response(self, user_input: str) -> str:
        """
        Get a response for the given user input by matching preprocessed input
        or using keywords to find possible responses.

        Args:
            user_input: The user's input message.

        Returns:
            The chatbot's response.
        """
        preprocessed_input = self.preprocess(user_input)
        if preprocessed_input in self.response_dict:
            return self.response_dict[preprocessed_input]
        else:
            keywords = self.get_keywords(preprocessed_input)
            possible_responses: list[str] = []
            for keyword in keywords:
                if keyword in self.keyword_dict:
                    possible_responses.extend(self.keyword_dict[keyword])
            if possible_responses:
                return max(set(possible_responses), key=possible_responses.count)
            else:
                return "I'm sorry, I didn't quite catch that. Could you please rephrase?"

    def run_chat_loop(self) -> None:
        """
        Run the chat loop, allowing the user to interact with the chatbot
        until they type 'exit' or 'quit'.
        """
        print("Bot: Hello! How can I assist you today? (Type 'exit' to quit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Bot: Goodbye!")
                break
            response = self.get_response(user_input)
            print(f"Bot: {response}")


if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.run_chat_loop()
