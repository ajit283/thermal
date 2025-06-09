# Thermal

## Description

Thermal is a Terminal User Interface (TUI) application for interacting with language models that adhere to the OpenAI API calling format. It allows you to chat with AI models directly from your terminal, stores conversation history locally in an SQLite database, and supports managing multiple conversations.

Thermal requires API access to a language model service that is compatible with the OpenAI API format. Configuration is primarily handled via an `endpoints.toml` file.

### Chat History Storage

Thermal stores your chat conversations in an SQLite database file named `chat_history.db`. The location of this file is determined as follows, in order of preference:

1.  In the `thermal` subdirectory of your user's data directory (e.g., `~/.local/share/thermal/chat_history.db` on Linux, or `~/Library/Application Support/thermal/chat_history.db` on macOS).
2.  If the data directory is not found, it tries the `thermal` subdirectory of your user's config directory (e.g., `~/.config/thermal/chat_history.db` on Linux).
3.  As a fallback, if neither of the above standard locations can be determined or accessed, the database will be created in the current working directory from which `thermal-cli` is launched (`./chat_history.db`).

### Endpoint Configuration

Thermal looks for an `endpoints.toml` configuration file in the following locations, in order:

1. The current directory (`./endpoints.toml`)
2. The XDG config directory (e.g., `~/.config/thermal/endpoints.toml` on Linux/macOS). If this directory doesn't exist, Thermal will attempt to create it.

If no `endpoints.toml` file is found, you will need to rely on environment variables for OpenAI (see below) or the application may not function correctly.

**`endpoints.toml` Structure:**

```toml
# Optional: Specify the default endpoint to use if multiple are defined.
# default_openai_endpoint = "my_default_openai"
# default_azure_endpoint = "my_default_azure"

[openai_endpoints.my_openai_config_name] # Replace 'my_openai_config_name' with your preferred name
api_key = "sk-yourOpenAIapiKey"            # Your API Key (optional if OPENAI_API_KEY env var is set)
api_base = "https://api.openai.com/v1"     # Optional: Defaults to OpenAI's standard API base
default_model = "gpt-4-turbo-preview"    # Specify the default model (e.g., gpt-4, gpt-3.5-turbo)

[azure_endpoints.my_azure_config_name]  # Replace 'my_azure_config_name' with your preferred name
api_key = "yourAzureOpenAIapiKey"         # Your Azure OpenAI API Key
api_base = "https://your-resource-name.openai.azure.com/" # Base URL of your Azure OpenAI resource
api_version = "2023-07-01-preview"      # API version for your Azure deployment
deployment_id = "yourDeploymentName"    # The name of your model deployment in Azure
```

**Key Configuration Details:**

- **Multiple Endpoints:** You can define multiple OpenAI and Azure endpoints, each with a unique name (e.g., `my_openai_config_name`, `my_azure_config_name`).
- **Default Endpoint:** Use `default_openai_endpoint` or `default_azure_endpoint` at the top level of the TOML file to specify which named configuration should be used by default if the application needs to pick one.
- **OpenAI `api_key`:** If `api_key` is not provided in the `[openai_endpoints.<name>]` section, Thermal (via the `async-openai` library) will likely attempt to use the `OPENAI_API_KEY` environment variable.
- **Azure Configuration:** For Azure, `api_key`, `api_base`, `api_version`, and `deployment_id` are crucial.

**Environment Variables:**

- `OPENAI_API_KEY`: Can be used as an alternative to specifying the `api_key` in the `endpoints.toml` file for OpenAI endpoints.

```bash
export OPENAI_API_KEY="sk-yourOpenAIapiKey"
```

## Keyboard Shortcuts

**General:**

- `Ctrl+C`: Quit the application (standard TUI behavior).

**Chatting Mode:**

_Input & Sending Messages:_

- **`Enter`**: Submit the current input text to the AI.
- Standard text input for typing your message (includes backspace, left/right arrow navigation).
- **`Ctrl+T`**: Opens the current text in the input box in an external editor (e.g., Vim, Nano). _Does not work if AI is responding._

_Navigation & Interaction:_

- **`Up Arrow`**: Selects the previous message in the chat history for viewing or editing.
- **`Down Arrow`**: Selects the next message in the chat history.
- **`Ctrl+N`**: Clears the current chat and starts a new, empty conversation. _Does not work if AI is responding._
- **`Ctrl+K`**: Switches to Conversation Picker mode to load or manage past conversations. _Does not work if AI is responding._
- **`Ctrl+E`**:
  - If AI is currently generating a response: Opens the _assistant's streaming response so far_ in an external editor (live view).
  - If a message is selected in the chat history: Opens the _content of the selected message_ in an external editor.
  - Otherwise (no message selected, AI not responding): Shows "No message selected".
- **`Ctrl+X`**: Attempts to cancel the current AI generation if one is in progress. Shows "Nothing to cancel" otherwise.

**Conversation Picker Mode:**

_Navigation & Selection:_

- **`Up Arrow`** or **`Shift+Tab`**: Moves the selection up in the list of conversations.
- **`Down Arrow`** or **`Tab`**: Moves the selection down in the list.
- **`Enter`**: Loads the selected conversation and switches back to Chatting Mode.

_Filtering & Management:_

- **Type to filter**: Characters typed will filter the conversation list.
- **`N`** (lowercase 'n'): Starts a new, empty chat session and switches to Chatting Mode.
- **`Ctrl+X`**: Deletes the currently selected conversation. A confirmation prompt will appear.

_Exiting Picker Mode:_

- **`Esc`**: Exits Conversation Picker mode and returns to Chatting Mode (clears filter text).
- **`Ctrl+K`**: (Alternative to `Esc`) Exits Conversation Picker mode (clears filter text).

## License

This project is licensed under the MIT License. See the `LICENSE.md` file for the full text.
