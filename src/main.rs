use async_openai::{
    Client,
    config::AzureConfig,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, Role,
    },
};
use crossterm::{
    cursor, // Import cursor module for Show/Hide
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use ratatui::{layout::Position, prelude::*, widgets::*};
use serde::Deserialize;
use std::process::Child;
use std::{
    collections::HashMap,
    env,
    error::Error,
    fs,
    io::{self, Write}, // Removed unused Stdout
    path::{Path, PathBuf},
    process::Command as StdCommand,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant as StdInstant},
};
use tempfile::Builder;
use tempfile::NamedTempFile;
// tempfile::NamedTempFile is not directly used but Builder::tempfile() returns it.
use tokio::sync::mpsc;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, Result as RusqliteResult, params};
use uuid::Uuid;

const APP_TITLE: &str = "Thermal";
const HIGHLIGHT_SYMBOL: &str = "> ";
const DB_FILE_NAME: &str = "chat_history.db";
const MAX_TITLE_LEN: usize = 50;

// --- Structs for TOML Configuration ---
#[derive(Deserialize, Debug, Default, Clone)]
struct EndpointsTomlConfig {
    default_openai_endpoint: Option<String>,
    default_azure_endpoint: Option<String>,
    openai_endpoints: Option<HashMap<String, OpenAIEndpointToml>>,
    azure_endpoints: Option<HashMap<String, AzureEndpointToml>>,
}

#[derive(Deserialize, Debug, Clone)]
struct OpenAIEndpointToml {
    api_key: Option<String>,
    api_base: Option<String>,
    default_model: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
struct AzureEndpointToml {
    api_key: Option<String>,
    api_base: Option<String>,
    api_version: Option<String>,
    deployment_id: Option<String>,
    default_model: Option<String>,
}

// --- load_toml_config function ---
fn load_toml_config(path: &Path) -> Result<EndpointsTomlConfig, Box<dyn Error>> {
    if !path.exists() {
        return Ok(EndpointsTomlConfig::default());
    }
    let content = std::fs::read_to_string(path)?;
    let config: EndpointsTomlConfig = toml::from_str(&content)?;
    Ok(config)
}

// --- Function to find endpoints.toml in multiple locations ---
fn find_toml_config_path() -> Result<PathBuf, Box<dyn Error>> {
    // First try current directory
    let current_dir_path = Path::new("endpoints.toml");
    if current_dir_path.exists() {
        return Ok(current_dir_path.to_path_buf());
    }

    // Then try XDG config directory
    if let Some(config_dir) = dirs::config_dir() {
        let thermal_config_dir = config_dir.join("thermal");
        let config_path = thermal_config_dir.join("endpoints.toml");

        if config_path.exists() {
            return Ok(config_path);
        }

        // If the file doesn't exist but we want to indicate where it should be created,
        // create the directory and return the path
        if let Err(e) = fs::create_dir_all(&thermal_config_dir) {
            eprintln!(
                "Warning: Could not create config directory {}: {}",
                thermal_config_dir.display(),
                e
            );
        }

        return Ok(config_path);
    }

    // Fallback to current directory if XDG config dir is not available
    Ok(current_dir_path.to_path_buf())
}

// --- Application State and Messages ---
#[derive(Clone)]
struct AppMessage {
    role: Role,
    content: String,
    timestamp: DateTime<Utc>,
    cached_wrapped_lines: Option<Vec<String>>, // Cache for wrapped lines
    cached_wrap_width: Option<u16>,            // Width used for the cached wrap
}

enum AppUpdate {
    AssistantChunk(String),
    AssistantError(String),
    AssistantDone,
    AssistantCancelled,
}

enum AppMode {
    Chatting,
    PickingConversation,
}

enum AppAction {
    None,
    LaunchEditor(String),
    LaunchStreamingEditor(String), // new "non-blocking" streaming editor
}

#[derive(Clone, Debug)]
struct ConversationMeta {
    id: String,
    title: String,
    updated_at: DateTime<Utc>,
}

// Create an enum to handle different client types
#[derive(Clone)]
enum ClientWrapper {
    OpenAI(Client<OpenAIConfig>),
    Azure(Client<AzureConfig>),
}

impl ClientWrapper {
    async fn chat_create_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<async_openai::types::ChatCompletionResponseStream, async_openai::error::OpenAIError>
    {
        match self {
            ClientWrapper::OpenAI(client) => client.chat().create_stream(request).await,
            ClientWrapper::Azure(client) => client.chat().create_stream(request).await,
        }
    }
}

struct App<'a> {
    input: String,
    messages: Vec<AppMessage>,
    openai_client: ClientWrapper,
    model_to_use: String,
    is_loading: bool,
    _config_source_message: String,
    update_sender: mpsc::Sender<AppUpdate>,
    update_receiver: mpsc::Receiver<AppUpdate>,
    selected_message: Option<usize>, // Replace message_list_state
    scroll_offset: usize,            // Add scroll offset for virtual scrolling
    input_cursor_position: usize,
    status_message: Option<String>,
    theme: Theme<'a>,
    db_conn: Connection,
    app_mode: AppMode,
    current_conversation_id: Option<String>,
    all_conversations: Vec<ConversationMeta>,
    picker_items: Vec<ConversationMeta>,
    picker_state: ListState,
    picker_filter_input: String,
    cancel_generation_flag: Arc<AtomicBool>,
    is_editing_current_input: bool,
    editor_file: Option<NamedTempFile>, // Will hold the temp file for streaming editor
    editor_process: Option<Child>,      // Will hold the child process for streaming editor
    is_external_editor_active: bool,    // Flag to indicate TUI suspension for external editor
    cached_conversation_title: Option<String>, // Cache for conversation title
}

struct Theme<'a> {
    base_style: Style,
    user_style: Style,
    assistant_style: Style,
    input_block_title_style: Style,
    chat_block_title_style: Style,
    loading_style: Style,
    status_style: Style,
    highlight_style: Style,
    picker_title_style: Style,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl Default for Theme<'_> {
    fn default() -> Self {
        // Simplified color scheme with just a couple of colors
        let base_bg = Color::Rgb(0x24, 0x27, 0x3a); // Catppuccin Macchiato base
        let text_fg = Color::Rgb(0xca, 0xd3, 0xf5); // Main text color

        // Primary accent color - soft purple
        let accent = Color::Rgb(0xc0, 0xb1, 0xed); // Soft purple

        // User and assistant both use the main text color
        let user_fg = text_fg;
        let assistant_fg = text_fg;

        // Use accent for titles and highlights
        let title_fg = accent;
        let loading_fg = accent;
        let status_fg = Color::Rgb(0xa6, 0xad, 0xc8); // Slightly dimmer for status

        // Highlight uses brighter text with subtle background
        let highlight_bg = Color::Rgb(0x35, 0x36, 0x45); // Slightly lighter than base background
        // Ensure highlighted text is very visible (white)
        let highlight_fg = Color::White;

        Theme {
            base_style: Style::default().fg(text_fg).bg(base_bg),
            user_style: Style::default().fg(user_fg),
            assistant_style: Style::default().fg(assistant_fg),
            input_block_title_style: Style::default().fg(title_fg),
            chat_block_title_style: Style::default().fg(title_fg),
            loading_style: Style::default().fg(loading_fg),
            status_style: Style::default().fg(status_fg),
            highlight_style: Style::default()
                .fg(highlight_fg)
                .bg(highlight_bg)
                .add_modifier(Modifier::BOLD),
            picker_title_style: Style::default().fg(title_fg).add_modifier(Modifier::BOLD),
            _phantom: std::marker::PhantomData,
        }
    }
}

// --- Database Functions ---
fn get_db_path() -> Result<PathBuf, Box<dyn Error>> {
    let mut path = dirs::data_dir()
        .ok_or_else(|| Box::<dyn Error>::from("Could not find user data directory"))?
        .join("ratatui_openai_chat");
    fs::create_dir_all(&path)?;
    path.push(DB_FILE_NAME);
    Ok(path)
}

fn initialize_database(conn: &Connection) -> RusqliteResult<()> {
    conn.execute_batch(
        "BEGIN;
        CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, timestamp TEXT NOT NULL, message_order INTEGER NOT NULL, FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations (updated_at);
        COMMIT;"
    )?;
    Ok(())
}

// --- Editor Function (blocking) ---
fn open_content_in_editor_blocking(initial_content: &str) -> Result<String, Box<dyn Error>> {
    let editor_env = env::var("EDITOR");
    let vis_env = env::var("VISUAL");

    let editor = vis_env.or(editor_env).unwrap_or_else(|_| {
        if cfg!(target_os = "windows") {
            "notepad".to_string()
        } else {
            "vi".to_string()
        }
    });

    let mut temp_file = Builder::new()
        .prefix("chat_edit_")
        .suffix(".md")
        .tempfile()?;

    temp_file.write_all(initial_content.as_bytes())?;
    temp_file.flush()?;

    let temp_file_path = temp_file.path().to_path_buf();
    let temp_file_path_str = temp_file_path.to_string_lossy();

    let mut command;
    if cfg!(target_os = "windows") {
        command = StdCommand::new("cmd");
        command.args(["/C", "start", "/WAIT", "\"\"", &editor, &temp_file_path_str]);
    } else {
        command = StdCommand::new("sh");
        let shell_command = format!("{} '{}'", editor, temp_file_path_str);
        command.arg("-c").arg(shell_command);
    }

    let status = command.status()?;

    if !status.success() {
        return Err(format!("Editor '{}' exited with status: {}", editor, status).into());
    }

    let edited_content = fs::read_to_string(&temp_file_path)?;
    Ok(edited_content)
}

impl<'a> App<'a> {
    fn new(
        client: ClientWrapper,
        model: String,
        config_source: String,
        db_conn: Connection,
    ) -> Self {
        let (tx, rx) = mpsc::channel(100);
        App {
            input: String::new(),
            messages: Vec::new(),
            openai_client: client,
            model_to_use: model,
            is_loading: false,
            _config_source_message: config_source,
            update_sender: tx,
            update_receiver: rx,
            selected_message: None,
            scroll_offset: 0,
            input_cursor_position: 0,
            status_message: Some("".to_string()),
            theme: Theme::default(),
            db_conn,
            app_mode: AppMode::Chatting,
            current_conversation_id: None,
            all_conversations: Vec::new(),
            picker_items: Vec::new(),
            picker_filter_input: String::new(),
            picker_state: ListState::default(),
            cancel_generation_flag: Arc::new(AtomicBool::new(false)),
            is_editing_current_input: false,
            editor_file: None,
            editor_process: None,
            is_external_editor_active: false, // Initialize new flag
            cached_conversation_title: None,  // Initialize cached conversation title
        }
    }

    fn create_new_conversation(
        &mut self,
        first_message_content: &str,
    ) -> Result<(), Box<dyn Error>> {
        let conv_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let title = first_message_content
            .chars()
            .take(MAX_TITLE_LEN)
            .collect::<String>();
        self.db_conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
            params![conv_id, title, now.to_rfc3339(), now.to_rfc3339()],
        )?;
        self.current_conversation_id = Some(conv_id);
        Ok(())
    }

    fn add_message_to_db(
        &self,
        conv_id: &str,
        role: Role,
        content: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<(), Box<dyn Error>> {
        let role_str = match role {
            Role::User => "user",
            Role::Assistant => "assistant",
            _ => "system",
        };
        let message_order: i64 = self.db_conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?1",
            params![conv_id],
            |row| row.get(0),
        )?;
        self.db_conn.execute("INSERT INTO messages (conversation_id, role, content, timestamp, message_order) VALUES (?1, ?2, ?3, ?4, ?5)", params![conv_id, role_str, content, timestamp.to_rfc3339(), message_order])?;
        self.db_conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), conv_id],
        )?;
        Ok(())
    }

    fn reset_message_cache(&mut self) {
        // Reset cached message formatting for all messages
        for msg in &mut self.messages {
            msg.cached_wrapped_lines = None;
            msg.cached_wrap_width = None;
        }
        // Reset cached conversation title
        self.cached_conversation_title = None;
    }

    fn submit_message(&mut self) {
        let trimmed_input = self.input.trim();
        if trimmed_input.is_empty() || self.is_loading {
            return;
        }

        self.cancel_generation_flag.store(false, Ordering::Relaxed);

        let user_message_content = trimmed_input.to_string();
        let now = Utc::now();
        if self.current_conversation_id.is_none() {
            if let Err(e) = self.create_new_conversation(&user_message_content) {
                self.status_message = Some(format!("Error creating conversation: {}", e));
                return;
            }
        }
        let current_conv_id = self.current_conversation_id.as_ref().unwrap().clone();
        self.messages.push(AppMessage {
            role: Role::User,
            content: user_message_content.clone(),
            timestamp: now,
            cached_wrapped_lines: None,
            cached_wrap_width: None,
        });
        if let Err(e) =
            self.add_message_to_db(&current_conv_id, Role::User, &user_message_content, now)
        {
            self.status_message = Some(format!("Error saving message: {}", e));
        }
        self.selected_message = Some(self.messages.len() - 1);
        self.input.clear();
        self.input_cursor_position = 0;
        self.is_loading = true;
        self.status_message = Some("Sending to AI...".to_string());

        // Clear the conversation title cache when a new message is submitted
        self.cached_conversation_title = None;

        let history_for_api: Vec<ChatCompletionRequestMessage> = self
            .messages
            .iter()
            .filter_map(|msg| match msg.role {
                Role::User => Some(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(msg.content.clone())
                        .build()
                        .unwrap()
                        .into(),
                ),
                Role::Assistant => Some(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(msg.content.clone())
                        .build()
                        .unwrap()
                        .into(),
                ),
                _ => None,
            })
            .collect();
        let client = self.openai_client.clone();
        let model = self.model_to_use.clone();
        let sender = self.update_sender.clone();
        let cancel_flag = self.cancel_generation_flag.clone();

        tokio::spawn(async move {
            let request_args = CreateChatCompletionRequestArgs::default()
                .model(&model)
                .messages(history_for_api)
                .stream(true)
                .build();

            if let Err(e) = request_args {
                let _ = sender
                    .send(AppUpdate::AssistantError(format!(
                        "Request build error: {}",
                        e
                    )))
                    .await;
                return;
            }

            let stream_result = client.chat_create_stream(request_args.unwrap()).await;

            match stream_result {
                Ok(mut stream) => {
                    let mut an_error_occurred_during_streaming = false;
                    while let Some(chunk_result) = stream.next().await {
                        if cancel_flag.load(Ordering::Relaxed) {
                            let _ = sender.send(AppUpdate::AssistantCancelled).await;
                            return;
                        }
                        match chunk_result {
                            Ok(response) => {
                                for choice in response.choices {
                                    if let Some(content) = choice.delta.content {
                                        if sender
                                            .send(AppUpdate::AssistantChunk(content))
                                            .await
                                            .is_err()
                                        {
                                            return;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = sender.send(AppUpdate::AssistantError(e.to_string())).await;
                                an_error_occurred_during_streaming = true;
                                break;
                            }
                        }
                    }

                    if an_error_occurred_during_streaming {
                        // Error already sent
                    } else {
                        let _ = sender.send(AppUpdate::AssistantDone).await;
                    }
                }
                Err(e) => {
                    let _ = sender
                        .send(AppUpdate::AssistantError(format!(
                            "Stream creation error: {}",
                            e
                        )))
                        .await;
                }
            }
        });
    }

    fn load_all_conversations_from_db(&mut self) -> Result<(), Box<dyn Error>> {
        let mut stmt = self.db_conn.prepare(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT 200",
        )?;
        let iter = stmt.query_map([], |row| {
            Ok(ConversationMeta {
                id: row.get(0)?,
                title: row.get(1)?,
                updated_at: row
                    .get::<_, String>(2)?
                    .parse::<DateTime<Utc>>()
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;

        self.all_conversations.clear();
        for item_result in iter {
            if let Ok(item) = item_result {
                self.all_conversations.push(item);
            } else if let Err(e) = item_result {
                self.status_message = Some(format!("Error loading a conversation: {}", e));
            }
        }
        Ok(())
    }

    fn apply_filter_and_update_picker_items(&mut self) {
        let matcher = SkimMatcherV2::default();

        // Get the currently selected index instead of ID
        let old_selected_idx = self.picker_state.selected();

        self.picker_items.clear();

        let mut matched_conversations: Vec<ConversationMeta> = Vec::new();
        if self.picker_filter_input.is_empty() {
            matched_conversations.extend_from_slice(&self.all_conversations);
        } else {
            for conv_meta in &self.all_conversations {
                if matcher
                    .fuzzy_match(&conv_meta.title, &self.picker_filter_input)
                    .is_some()
                {
                    matched_conversations.push(conv_meta.clone());
                }
            }
        }
        self.picker_items.extend(matched_conversations);

        // Always select the first item when filter changes
        if !self.picker_items.is_empty() {
            self.picker_state.select(Some(0));
        } else {
            self.picker_state.select(None);
        }
    }

    fn handle_chatting_input(
        &mut self,
        key_code: KeyCode,
        key_modifiers: KeyModifiers,
    ) -> AppAction {
        if key_modifiers == KeyModifiers::CONTROL {
            match key_code {
                KeyCode::Char('n') => {
                    if self.is_loading {
                        self.status_message =
                            Some("Cannot start new chat while AI is responding.".to_string());
                        return AppAction::None;
                    }
                    // Clear current chat and start a new one
                    self.messages.clear();
                    self.current_conversation_id = None;
                    self.input.clear();
                    self.input_cursor_position = 0;
                    self.selected_message = None;
                    // Reset caches for new chat
                    self.reset_message_cache();
                    self.status_message = Some("New chat started.".to_string());
                    return AppAction::None;
                }
                KeyCode::Char('k') => {
                    if self.is_loading {
                        self.status_message =
                            Some("Cannot open picker while AI is responding.".to_string());
                        return AppAction::None;
                    }
                    self.app_mode = AppMode::PickingConversation;
                    self.picker_filter_input.clear();
                    // Reset the cached conversation title when changing conversations
                    self.cached_conversation_title = None;
                    if let Err(e) = self.load_all_conversations_from_db() {
                        self.status_message = Some(format!("Error loading conversations: {}", e));
                        self.app_mode = AppMode::Chatting;
                    } else {
                        self.apply_filter_and_update_picker_items();
                    }
                    return AppAction::None;
                }
                KeyCode::Char('e') => {
                    if self.is_loading {
                        if let Some(last) =
                            self.messages.last().filter(|m| m.role == Role::Assistant)
                        {
                            self.status_message = Some("Opening live editor…".into());
                            return AppAction::LaunchStreamingEditor(last.content.clone());
                        } else {
                            self.status_message = Some("No assistant text to edit".into());
                            return AppAction::None;
                        }
                    } else {
                        let content_to_edit = if let Some(idx) = self.selected_message {
                            self.messages.get(idx).map(|m| m.content.clone())
                        } else {
                            None
                        };
                        if let Some(c) = content_to_edit {
                            self.status_message = Some("Opening message in editor…".into());
                            return AppAction::LaunchEditor(c);
                        } else {
                            self.status_message = Some("No message selected".into());
                            return AppAction::None;
                        }
                    }
                }
                KeyCode::Char('t') => {
                    if self.is_loading {
                        self.status_message =
                            Some("Cannot edit input while AI is responding.".to_string());
                        return AppAction::None;
                    }
                    self.is_editing_current_input = true;
                    self.status_message = Some("Opening input in editor...".to_string());
                    return AppAction::LaunchEditor(self.input.clone());
                }
                KeyCode::Char('x') => {
                    if self.is_loading {
                        self.cancel_generation_flag.store(true, Ordering::Relaxed);
                        self.status_message =
                            Some("Attempting to cancel generation...".to_string());
                    } else {
                        self.status_message = Some("Nothing to cancel.".to_string());
                    }
                    return AppAction::None;
                }
                _ => {}
            }
        }

        match key_code {
            KeyCode::Enter => self.submit_message(),
            KeyCode::Char(c) => {
                self.input.insert(self.input_cursor_position, c);
                self.input_cursor_position += 1;
            }
            KeyCode::Backspace => {
                if self.input_cursor_position > 0 {
                    self.input_cursor_position -= 1;
                    self.input.remove(self.input_cursor_position);
                }
            }
            KeyCode::Left => {
                if self.input_cursor_position > 0 {
                    self.input_cursor_position -= 1;
                }
            }
            KeyCode::Right => {
                if self.input_cursor_position < self.input.len() {
                    self.input_cursor_position += 1;
                }
            }
            KeyCode::Up => {
                if !self.messages.is_empty() {
                    match self.selected_message {
                        Some(idx) if idx > 0 => self.selected_message = Some(idx - 1),
                        Some(_) => self.selected_message = Some(0),
                        None => self.selected_message = Some(self.messages.len() - 1),
                    }
                }
            }
            KeyCode::Down => {
                if !self.messages.is_empty() {
                    match self.selected_message {
                        Some(idx) if idx < self.messages.len() - 1 => {
                            self.selected_message = Some(idx + 1)
                        }
                        Some(idx) => self.selected_message = Some(idx),
                        None => self.selected_message = Some(0),
                    }
                }
            }
            _ => {}
        }
        AppAction::None
    }

    fn handle_picker_input(&mut self, key_code: KeyCode, key_modifiers: KeyModifiers) {
        match key_code {
            KeyCode::Esc => {
                self.app_mode = AppMode::Chatting;
                self.picker_filter_input.clear();
            }
            KeyCode::Up | KeyCode::BackTab => {
                if !self.picker_items.is_empty() {
                    let current = self.picker_state.selected().unwrap_or(0);
                    self.picker_state.select(Some(current.saturating_sub(1)));
                }
            }
            KeyCode::Down | KeyCode::Tab => {
                if !self.picker_items.is_empty() {
                    let current = self.picker_state.selected().unwrap_or(0);
                    if current < self.picker_items.len() - 1 {
                        self.picker_state.select(Some(current + 1));
                    }
                }
            }
            KeyCode::Char('k') if key_modifiers == KeyModifiers::CONTROL => {
                self.app_mode = AppMode::Chatting;
                self.picker_filter_input.clear();
            }
            KeyCode::Enter => {
                if let Some(idx) = self.picker_state.selected() {
                    if idx < self.picker_items.len() {
                        let item = self.picker_items[idx].clone();
                        if let Err(e) = self.load_selected_conversation(&item.id) {
                            self.status_message = Some(format!("Err load chat: {}", e));
                        } else {
                            self.status_message = Some(format!("Loaded: {}", item.title));
                            // Reset caches for newly loaded conversation
                            self.reset_message_cache();
                        }
                        self.app_mode = AppMode::Chatting;
                        self.picker_filter_input.clear();
                    }
                }
            }
            KeyCode::Char('n') => {
                self.messages.clear();
                self.current_conversation_id = None;
                self.input.clear();
                self.selected_message = None;
                self.status_message = Some("New chat started.".to_string());
                // Reset caches for new chat
                self.reset_message_cache();
                self.app_mode = AppMode::Chatting;
                self.picker_filter_input.clear();
            }
            KeyCode::Char(c) if c == 'x' && key_modifiers == KeyModifiers::CONTROL => {
                if let Some(selected_idx) = self.picker_state.selected() {
                    if selected_idx < self.picker_items.len() {
                        let item_to_delete = self.picker_items[selected_idx].clone();

                        match self.db_conn.execute(
                            "DELETE FROM conversations WHERE id = ?1",
                            params![item_to_delete.id],
                        ) {
                            Ok(num_deleted) => {
                                if num_deleted > 0 {
                                    self.status_message =
                                        Some(format!("Deleted: {}", item_to_delete.title));
                                    if let Err(e) = self.load_all_conversations_from_db() {
                                        self.status_message = Some(format!(
                                            "Error reloading conversations after delete: {}",
                                            e
                                        ));
                                    }
                                    self.apply_filter_and_update_picker_items();
                                } else {
                                    self.status_message = Some(format!(
                                        "Could not delete: '{}' (not found in DB).",
                                        item_to_delete.title
                                    ));
                                    if let Err(e) = self.load_all_conversations_from_db() {
                                        self.status_message =
                                            Some(format!("Error reloading conversations: {}", e));
                                    }
                                    self.apply_filter_and_update_picker_items();
                                }
                            }
                            Err(e) => {
                                self.status_message = Some(format!(
                                    "Error deleting '{}': {}",
                                    item_to_delete.title, e
                                ));
                            }
                        }
                    }
                }
            }
            KeyCode::Char(c) => {
                self.picker_filter_input.push(c);
                self.apply_filter_and_update_picker_items();
            }
            KeyCode::Backspace => {
                self.picker_filter_input.pop();
                self.apply_filter_and_update_picker_items();
            }
            _ => {}
        }
    }

    fn load_selected_conversation(&mut self, conv_id: &str) -> Result<(), Box<dyn Error>> {
        self.messages.clear();
        let mut stmt = self.db_conn.prepare("SELECT role, content, timestamp FROM messages WHERE conversation_id = ?1 ORDER BY message_order ASC")?;
        let iter = stmt.query_map(params![conv_id], |row| {
            let role_str: String = row.get(0)?;
            let role = match role_str.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => Role::System,
            };
            Ok(AppMessage {
                role,
                content: row.get(1)?,
                timestamp: row
                    .get::<_, String>(2)?
                    .parse::<DateTime<Utc>>()
                    .unwrap_or_else(|_| Utc::now()),
                cached_wrapped_lines: None,
                cached_wrap_width: None,
            })
        })?;
        for msg_result in iter {
            if let Ok(msg) = msg_result {
                self.messages.push(msg);
            } else if let Err(e) = msg_result {
                self.status_message = Some(format!("Err load msg: {}", e));
            }
        }
        self.current_conversation_id = Some(conv_id.to_string());
        self.selected_message = if self.messages.is_empty() {
            None
        } else {
            Some(self.messages.len() - 1)
        };

        // Clear the conversation title cache when loading a new conversation
        self.cached_conversation_title = None;
        Ok(())
    }

    fn update_from_channel(&mut self) -> bool {
        let mut did_receive_update = false;

        // Process ALL available messages in the channel, not just one
        loop {
            match self.update_receiver.try_recv() {
                Ok(AppUpdate::AssistantChunk(chunk)) => {
                    did_receive_update = true;
                    let mut is_new_msg = false;
                    if self
                        .messages
                        .last()
                        .map_or(true, |m| m.role != Role::Assistant)
                        || self.messages.is_empty()
                    {
                        if !chunk.is_empty() {
                            self.messages.push(AppMessage {
                                role: Role::Assistant,
                                content: chunk.clone(),
                                timestamp: Utc::now(),
                                cached_wrapped_lines: None,
                                cached_wrap_width: None,
                            });
                            is_new_msg = true;
                        }
                    } else if let Some(last) = self.messages.last_mut() {
                        if last.role == Role::Assistant {
                            last.content.push_str(&chunk);
                            // Invalidate cache when content changes
                            last.cached_wrapped_lines = None;
                            last.cached_wrap_width = None;
                        } else if !chunk.is_empty() {
                            self.messages.push(AppMessage {
                                role: Role::Assistant,
                                content: chunk.clone(),
                                timestamp: Utc::now(),
                                cached_wrapped_lines: None,
                                cached_wrap_width: None,
                            });
                            is_new_msg = true;
                        }
                    }

                    if !self.is_external_editor_active {
                        self.status_message = Some("AI Typing...".to_string());
                        // Always keep the latest message selected when AI is typing
                        if !self.messages.is_empty() {
                            self.selected_message = Some(self.messages.len() - 1);
                        }
                    }

                    if self.is_external_editor_active {
                        // Check this instead of self.editor_file.is_some()
                        if let Some(file) = &mut self.editor_file {
                            if let Err(e) = file.write_all(chunk.as_bytes()) {
                                eprintln!("Error writing to streaming editor file: {}", e);
                            } else if let Err(e) = file.flush() {
                                eprintln!("Error flushing streaming editor file: {}", e);
                            }
                        }
                    }
                }
                Ok(AppUpdate::AssistantError(err_msg)) => {
                    did_receive_update = true;
                    self.is_loading = false;
                    let content = format!("[Error]: {}", err_msg);
                    let now = Utc::now();
                    self.messages.push(AppMessage {
                        role: Role::Assistant,
                        content: content.clone(),
                        timestamp: now,
                        cached_wrapped_lines: None,
                        cached_wrap_width: None,
                    });
                    if let Some(id) = &self.current_conversation_id {
                        if let Err(e) = self.add_message_to_db(id, Role::Assistant, &content, now) {
                            if !self.is_external_editor_active {
                                self.status_message = Some(format!("Err save err: {}", e));
                            } else {
                                eprintln!("Err save err: {}", e);
                            }
                        }
                    }

                    if !self.is_external_editor_active {
                        self.status_message = Some(format!(
                            "Error: {}",
                            err_msg.chars().take(50).collect::<String>()
                        ));
                        if !self.messages.is_empty() {
                            self.selected_message = Some(self.messages.len() - 1);
                        }
                    } else {
                        eprintln!("Assistant Error while editor active: {}", err_msg);
                    }

                    if self.is_external_editor_active {
                        // If editor was active, try to clean up
                        if let Some(mut child) = self.editor_process.take() {
                            let _ = child.kill(); // Attempt to kill editor
                            let _ = child.wait(); // Wait for it to ensure cleanup
                        }
                        self.editor_file = None; // This will drop and delete the temp file
                        // self.is_external_editor_active will be reset by run_app's check
                    }
                }
                Ok(AppUpdate::AssistantDone) => {
                    did_receive_update = true;
                    self.is_loading = false;
                    if let Some(id) = &self.current_conversation_id {
                        if let Some(last) = self.messages.last() {
                            if last.role == Role::Assistant && !last.content.is_empty() {
                                if let Err(e) = self.add_message_to_db(
                                    id,
                                    Role::Assistant,
                                    &last.content,
                                    last.timestamp,
                                ) {
                                    if !self.is_external_editor_active {
                                        self.status_message =
                                            Some(format!("Err save AI resp: {}", e));
                                    } else {
                                        eprintln!("Err save AI resp: {}", e);
                                    }
                                }
                            }
                        }
                    }

                    if !self.is_external_editor_active {
                        self.status_message = Some("Ready.".to_string());
                        if self
                            .messages
                            .last()
                            .map_or(false, |m| m.role == Role::Assistant && m.content.is_empty())
                        {
                            self.messages.pop();
                        }
                        if !self.messages.is_empty() {
                            self.selected_message = Some(self.messages.len() - 1);
                        } else {
                            self.selected_message = None;
                        }
                    }
                    // If editor was active, user will close it. `run_app` handles process exit.
                }
                Ok(AppUpdate::AssistantCancelled) => {
                    did_receive_update = true;
                    self.is_loading = false;
                    let mut final_status_message = "AI generation cancelled.".to_string();

                    if let Some(last_msg_idx) = self.messages.len().checked_sub(1) {
                        if self.messages[last_msg_idx].role == Role::Assistant {
                            if !self.messages[last_msg_idx].content.is_empty() {
                                let content_to_save = self.messages[last_msg_idx].content.clone();
                                let timestamp_to_save = self.messages[last_msg_idx].timestamp;
                                if let Some(conv_id) = &self.current_conversation_id {
                                    match self.add_message_to_db(
                                        conv_id,
                                        Role::Assistant,
                                        &content_to_save,
                                        timestamp_to_save,
                                    ) {
                                        Ok(_) => {
                                            final_status_message =
                                                "AI generation cancelled. Partial response saved."
                                                    .to_string();
                                        }
                                        Err(e) => {
                                            final_status_message = format!(
                                                "AI cancelled. Error saving partial response: {}",
                                                e
                                            );
                                        }
                                    }
                                } else {
                                    final_status_message =
                                        "AI cancelled. No active conv (partial content not saved)."
                                            .to_string();
                                }
                            } else {
                                self.messages.pop(); // Remove empty assistant message
                                final_status_message =
                                    "AI generation cancelled. No content generated.".to_string();
                            }
                        }
                    }

                    if !self.is_external_editor_active {
                        self.status_message = Some(final_status_message);
                        if self.messages.is_empty() {
                            self.selected_message = None;
                        } else {
                            self.selected_message = Some(self.messages.len() - 1);
                        }
                    } else {
                        eprintln!("{}", final_status_message);
                        // If editor was active, user will close it. `run_app` handles process exit.
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    // No more messages available, break the loop
                    break;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    did_receive_update = true;
                    self.is_loading = false;
                    if !self.is_external_editor_active {
                        self.status_message = Some("AI worker connection lost.".to_string());
                    } else {
                        eprintln!("AI worker connection lost.");
                    }
                    if self.is_external_editor_active {
                        // If editor was active, try to clean up
                        if let Some(mut child) = self.editor_process.take() {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                        self.editor_file = None;
                    }
                    break;
                }
            }
        }

        // Return whether we received any updates
        return did_receive_update;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let mut chosen_endpoint_name_from_cli: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--endpoint" {
            if i + 1 < args.len() {
                chosen_endpoint_name_from_cli = Some(args[i + 1].clone());
                break;
            } else {
                eprintln!("Warning: --endpoint flag provided without a value.");
            }
        }
        i += 1;
    }
    let toml_path = find_toml_config_path().unwrap_or_else(|e| {
        eprintln!(
            "Warning: Could not determine config path: {}. Using current directory.",
            e
        );
        PathBuf::from("endpoints.toml")
    });

    println!("Looking for config at: {}", toml_path.display());

    let toml_config = load_toml_config(&toml_path).unwrap_or_else(|err| {
        if toml_path.exists() {
            eprintln!(
                "Warning: Could not parse '{}': {}. Using defaults.",
                toml_path.display(),
                err
            );
        } else {
            println!(
                "Config file '{}' not found. Using defaults.",
                toml_path.display()
            );
        }
        EndpointsTomlConfig::default()
    });
    let mut selected_openai_endpoint: Option<OpenAIEndpointToml> = None;
    let mut selected_azure_endpoint: Option<AzureEndpointToml> = None;
    let mut config_source_message: String = "Defaults (e.g. OPENAI_API_KEY env var)".to_string();

    // First, try to find the endpoint by name in both OpenAI and Azure sections
    if let Some(name) = &chosen_endpoint_name_from_cli {
        // Check OpenAI endpoints first
        if let Some(openai_endpoints_map) = &toml_config.openai_endpoints {
            if let Some(details) = openai_endpoints_map.get(name) {
                selected_openai_endpoint = Some(details.clone());
                config_source_message = format!("OpenAI endpoint '{}' from TOML", name);
            }
        }

        // If not found in OpenAI, check Azure endpoints
        if selected_openai_endpoint.is_none() {
            if let Some(azure_endpoints_map) = &toml_config.azure_endpoints {
                if let Some(details) = azure_endpoints_map.get(name) {
                    selected_azure_endpoint = Some(details.clone());
                    config_source_message = format!("Azure endpoint '{}' from TOML", name);
                }
            }
        }

        if selected_openai_endpoint.is_none() && selected_azure_endpoint.is_none() {
            eprintln!("Warning: Endpoint '{}' not found in TOML.", name);
        }
    }

    // If no CLI endpoint specified or not found, try defaults
    if selected_openai_endpoint.is_none() && selected_azure_endpoint.is_none() {
        // Try default Azure endpoint first
        if let Some(default_azure_name) = &toml_config.default_azure_endpoint {
            if let Some(azure_endpoints_map) = &toml_config.azure_endpoints {
                if let Some(details) = azure_endpoints_map.get(default_azure_name) {
                    selected_azure_endpoint = Some(details.clone());
                    config_source_message =
                        format!("default Azure endpoint '{}' from TOML", default_azure_name);
                }
            }
        }

        // If no Azure default, try OpenAI default
        if selected_azure_endpoint.is_none() {
            if let Some(default_openai_name) = &toml_config.default_openai_endpoint {
                if let Some(openai_endpoints_map) = &toml_config.openai_endpoints {
                    if let Some(details) = openai_endpoints_map.get(default_openai_name) {
                        selected_openai_endpoint = Some(details.clone());
                        config_source_message = format!(
                            "default OpenAI endpoint '{}' from TOML",
                            default_openai_name
                        );
                    }
                }
            }
        }
    }

    // If still no endpoints found, try the first available endpoint
    if selected_openai_endpoint.is_none() && selected_azure_endpoint.is_none() {
        // Try first Azure endpoint
        if let Some(azure_endpoints_map) = &toml_config.azure_endpoints {
            if let Some((name, details)) = azure_endpoints_map.iter().next() {
                selected_azure_endpoint = Some(details.clone());
                config_source_message =
                    format!("first available Azure endpoint '{}' from TOML", name);
            }
        }

        // If no Azure endpoints, try first OpenAI endpoint
        if selected_azure_endpoint.is_none() {
            if let Some(openai_endpoints_map) = &toml_config.openai_endpoints {
                if let Some((name, details)) = openai_endpoints_map.iter().next() {
                    selected_openai_endpoint = Some(details.clone());
                    config_source_message =
                        format!("first available OpenAI endpoint '{}' from TOML", name);
                }
            }
        }
    }

    let mut model_to_use = "gpt-4o".to_string(); // Default model

    // Determine client type and create appropriate client
    let client = if let Some(azure_details) = &selected_azure_endpoint {
        // Create Azure client
        let mut azure_config = AzureConfig::new();

        if let Some(key) = &azure_details.api_key {
            azure_config = azure_config.with_api_key(key);
        }
        if let Some(base) = &azure_details.api_base {
            azure_config = azure_config.with_api_base(base);
        }
        if let Some(version) = &azure_details.api_version {
            azure_config = azure_config.with_api_version(version);
        }
        if let Some(deployment) = &azure_details.deployment_id {
            azure_config = azure_config.with_deployment_id(deployment);
        }

        if let Some(model) = &azure_details.default_model {
            model_to_use = model.clone();
        }

        ClientWrapper::Azure(Client::with_config(azure_config))
    } else if let Some(openai_details) = &selected_openai_endpoint {
        // Create OpenAI client
        let mut openai_client_config = OpenAIConfig::new();

        if let Some(key) = &openai_details.api_key {
            openai_client_config = openai_client_config.with_api_key(key);
        }
        if let Some(base) = &openai_details.api_base {
            openai_client_config = openai_client_config.with_api_base(base);
        }
        if let Some(model) = &openai_details.default_model {
            model_to_use = model.clone();
        }

        ClientWrapper::OpenAI(Client::with_config(openai_client_config))
    } else {
        // Default OpenAI client
        let openai_client_config = OpenAIConfig::new();
        ClientWrapper::OpenAI(Client::with_config(openai_client_config))
    };
    println!(
        "OpenAI client: {}. Model: {}.",
        &config_source_message, &model_to_use
    );
    let db_path = get_db_path()?;
    println!("DB at: {}", db_path.display());
    let db_conn = Connection::open(&db_path)
        .map_err(|e| format!("DB open error {}: {}", db_path.display(), e))?;
    initialize_database(&db_conn).map_err(|e| format!("DB init error: {}", e))?;
    println!("DB initialized.");
    println!("Starting TUI...");

    // Ensure TUI setup only happens if no editor is immediately launched before main loop
    // For now, setup is standard as editor launch is event-driven.
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?; // Mouse capture might not be needed if not used
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(client, model_to_use, config_source_message, db_conn);

    let res = run_app(&mut terminal, &mut app).await;

    // Cleanup TUI settings
    disable_raw_mode()?;
    execute!(
        io::stdout(), // Use a fresh handle if terminal's stdout was moved
        LeaveAlternateScreen,
        DisableMouseCapture, // Ensure mouse capture is disabled
        cursor::Show         // Always show cursor on exit
    )?;
    // terminal.show_cursor()?; // Already handled by execute! above

    if let Err(err) = res {
        eprintln!("App error: {:?}", err);
    }
    Ok(())
}

async fn run_app<B: Backend + std::io::Write>(
    terminal: &mut Terminal<B>,
    app: &mut App<'_>,
) -> io::Result<()> {
    // Use a faster tick rate for better responsiveness
    let tick_rate = Duration::from_millis(50); // Reduced from 250ms to 50ms
    let mut last_tick = StdInstant::now();
    let mut content_for_blocking_editor: Option<String> = None;
    let mut needs_redraw = true; // Flag to track if UI needs redrawing

    // Force initial draw
    terminal.draw(|f| ui(f, app))?;

    loop {
        // Process AI updates and set redraw flag if updates occurred
        if app.update_from_channel() {
            needs_redraw = true;
        }

        // --- Streaming Editor Management ---
        if app.editor_process.is_some() {
            // Actively manage if a process was spawned
            let mut editor_finished_this_tick = false;
            if let Some(editor_process_mut) = &mut app.editor_process {
                match editor_process_mut.try_wait() {
                    Ok(Some(_exit_status)) => {
                        // Editor exited
                        editor_finished_this_tick = true;
                        needs_redraw = true;
                    }
                    Ok(None) => {
                        // Editor still running
                        app.is_external_editor_active = true;
                        // Minimal sleep to prevent busy-looping, still allow AI chunks to be processed
                        tokio::time::sleep(Duration::from_millis(30)).await; // shorter sleep
                        continue; // Skip TUI drawing and TUI event polling
                    }
                    Err(e) => {
                        // Error checking process status
                        eprintln!("Error checking editor process: {}", e);
                        editor_finished_this_tick = true; // Treat as finished to attempt cleanup
                        needs_redraw = true;
                    }
                }
            }

            if editor_finished_this_tick {
                app.editor_process = None; // Clear the process
                app.editor_file = None; // Drop NamedTempFile, which deletes the file
                app.is_external_editor_active = false;
                needs_redraw = true;

                // Restore TUI
                enable_raw_mode()?;
                execute!(
                    terminal.backend_mut(), // Use terminal's backend
                    EnterAlternateScreen,
                    EnableMouseCapture,
                    cursor::Hide
                )?;
                terminal.clear()?; // Clear editor remnants from screen
                app.status_message = Some("Streaming editor closed. Resuming TUI.".to_string());
                terminal.draw(|f| ui(f, app))?; // Force redraw TUI
                last_tick = StdInstant::now();
                needs_redraw = false; // Just redrew
            }
        } else {
            // If editor_process is None, ensure is_external_editor_active is false.
            if app.is_external_editor_active {
                app.is_external_editor_active = false;
                needs_redraw = true;
                // This state implies an unexpected editor closure or cleanup.
                // Ensure TUI is restored.
                enable_raw_mode()?;
                execute!(
                    terminal.backend_mut(),
                    EnterAlternateScreen,
                    EnableMouseCapture,
                    cursor::Hide
                )?;
                terminal.clear()?;
                app.status_message = Some("Editor session ended. Resuming TUI.".to_string());
                terminal.draw(|f| ui(f, app))?;
                last_tick = StdInstant::now();
                needs_redraw = false; // Just redrew
            }
        }

        // If an external editor is active, skip the rest of the loop
        if app.is_external_editor_active {
            continue;
        }

        // --- Standard TUI Processing (only if no external editor is active) ---
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        // Event handling - only poll if timeout hasn't been reached
        if crossterm::event::poll(timeout)? {
            let event = event::read()?;
            match event {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    needs_redraw = true; // Any key press generally requires a redraw

                    if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c') {
                        if let Some(mut child) = app.editor_process.take() {
                            // if streaming editor is running
                            let _ = child.kill(); // Attempt to kill it
                            let _ = child.wait(); // Wait for it
                        }
                        app.editor_file = None; // Ensure temp file is dropped/deleted
                        return Ok(()); // Exit app
                    }

                    let action_result = match app.app_mode {
                        AppMode::Chatting => app.handle_chatting_input(key.code, key.modifiers),
                        AppMode::PickingConversation => {
                            app.handle_picker_input(key.code, key.modifiers);
                            AppAction::None
                        }
                    };

                    match action_result {
                        AppAction::LaunchEditor(content) => {
                            content_for_blocking_editor = Some(content);
                            // app.is_editing_current_input is set within handle_chatting_input
                        }
                        AppAction::LaunchStreamingEditor(initial_content) => {
                            // 1. Suspend TUI
                            disable_raw_mode()?; // Give raw mode control to editor
                            execute!(
                                terminal.backend_mut(),
                                LeaveAlternateScreen,
                                DisableMouseCapture,
                                cursor::Show // Show cursor for editor
                            )?;

                            // 2. Create temp file
                            match Builder::new()
                                .prefix("chat_stream_")
                                .suffix(".md")
                                .tempfile()
                            {
                                Ok(mut file) => {
                                    if file.write_all(initial_content.as_bytes()).is_err()
                                        || file.flush().is_err()
                                    {
                                        // Restore TUI immediately on file error
                                        app.status_message = Some(
                                            "Streaming editor file write/flush failed.".into(),
                                        );
                                        enable_raw_mode()?;
                                        execute!(
                                            terminal.backend_mut(),
                                            EnterAlternateScreen,
                                            EnableMouseCapture,
                                            cursor::Hide
                                        )?;
                                        terminal.clear()?; // Clear any partial TUI suspension artifacts
                                        needs_redraw = true;
                                    } else {
                                        let path_str = file.path().to_string_lossy().to_string();
                                        app.editor_file = Some(file); // Keep file alive in App

                                        // 3. Launch editor (non-blocking spawn)
                                        let editor_name = env::var("VISUAL")
                                            .or_else(|_| env::var("EDITOR"))
                                            .unwrap_or_else(|_| {
                                                if cfg!(target_os = "windows") {
                                                    "notepad".to_string()
                                                } else {
                                                    "vi".to_string()
                                                }
                                            });

                                        let mut cmd;
                                        if cfg!(target_os = "windows") {
                                            cmd = StdCommand::new(&editor_name);
                                            cmd.arg(&path_str);
                                        } else {
                                            cmd = StdCommand::new(&editor_name);
                                            cmd.arg(&path_str);
                                        }

                                        match cmd.spawn() {
                                            Ok(child) => {
                                                app.editor_process = Some(child);
                                                app.is_external_editor_active = true; // TUI is now effectively suspended
                                                eprintln!(
                                                    "Launched streaming editor: {} with file {}",
                                                    editor_name, path_str
                                                );
                                            }
                                            Err(e) => {
                                                app.status_message = Some(format!(
                                                    "Failed to launch streaming editor: {}",
                                                    e
                                                ));
                                                app.editor_file = None; // Clean up temp file if spawn fails
                                                // Restore TUI
                                                enable_raw_mode()?;
                                                execute!(
                                                    terminal.backend_mut(),
                                                    EnterAlternateScreen,
                                                    EnableMouseCapture,
                                                    cursor::Hide
                                                )?;
                                                terminal.clear()?;
                                                needs_redraw = true;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    // Tempfile creation failed
                                    app.status_message = Some(format!(
                                        "Failed to create streaming editor temp file: {}",
                                        e
                                    ));
                                    // Restore TUI (though it might not have been fully suspended yet)
                                    enable_raw_mode()?; // Ensure raw mode is on for TUI
                                    execute!(
                                        terminal.backend_mut(),
                                        EnterAlternateScreen,
                                        EnableMouseCapture,
                                        cursor::Hide
                                    )?;
                                    terminal.clear()?;
                                    needs_redraw = true;
                                }
                            }
                        }
                        AppAction::None => {}
                    }
                }
                Event::Resize(_, _) => {
                    // Terminal was resized, force a redraw
                    needs_redraw = true;
                    // Reset message cache to recalculate wrapping
                    app.reset_message_cache();
                }
                _ => {}
            }
        } else if last_tick.elapsed() >= tick_rate {
            // No event, but tick timeout reached
            last_tick = StdInstant::now();
            // Only redraw on tick if there's an animation (like AI typing)
            if app.is_loading {
                needs_redraw = true;
            }
        }

        // --- Blocking Editor Launch (Ctrl+T or Ctrl+E when not streaming) ---
        if let Some(text_to_edit) = content_for_blocking_editor.take() {
            // Suspend TUI
            disable_raw_mode()?;
            execute!(
                terminal.backend_mut(),
                LeaveAlternateScreen,
                DisableMouseCapture,
                cursor::Show
            )?;

            let editor_result = open_content_in_editor_blocking(&text_to_edit);

            // Resume TUI
            enable_raw_mode()?;
            execute!(
                terminal.backend_mut(),
                EnterAlternateScreen,
                EnableMouseCapture,
                cursor::Hide
            )?;
            terminal.clear()?; // Clear editor remnants

            match editor_result {
                Ok(edited_text) => {
                    if app.is_editing_current_input {
                        app.input = edited_text;
                        app.input_cursor_position = app.input.len();
                        app.status_message = Some("Input updated from editor.".to_string());
                    } else {
                        let mut preview = edited_text.chars().take(50).collect::<String>();
                        if edited_text.chars().count() > 50 {
                            preview.push_str("...");
                        }
                        app.status_message =
                            Some(format!("Editor closed. Content (viewed): \"{}\"", preview));
                    }
                }
                Err(e) => {
                    app.status_message = Some(format!("Editor action failed: {}", e));
                }
            }
            app.is_editing_current_input = false; // Reset flag
            needs_redraw = true;
        }

        // Only redraw if needed
        if needs_redraw && !app.is_external_editor_active {
            terminal.draw(|f| ui(f, app))?;
            needs_redraw = false;
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    f.render_widget(Clear, f.area()); // Clear entire frame first
    match app.app_mode {
        AppMode::Chatting => ui_chatting(f, app),
        AppMode::PickingConversation => ui_picker(f, app),
    }
}

fn ui_chatting(f: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Min(0),    // Chat messages
                Constraint::Length(1), // Status bar
                Constraint::Length(3), // Input box
            ]
            .as_ref(),
        )
        .split(f.area());

    // Render title bar
    let title_block = Block::default()
        .borders(Borders::NONE)
        .title(Span::styled(
            APP_TITLE,
            theme.chat_block_title_style.clone(),
        ))
        .title_alignment(Alignment::Center);
    f.render_widget(title_block, chunks[0]);

    // Calculate visible messages area (subtract title line)
    let messages_area = Rect {
        x: chunks[0].x,
        y: chunks[0].y + 1,
        width: chunks[0].width,
        height: chunks[0].height.saturating_sub(1),
    };

    // Clone theme styles needed for message rendering to avoid borrowing conflict
    let user_style = theme.user_style;
    let assistant_style = theme.assistant_style;
    let highlight_style = theme.highlight_style;
    let loading_style = theme.loading_style;
    let status_style = theme.status_style;
    let base_style = theme.base_style;
    let input_block_title_style = theme.input_block_title_style.clone();

    render_messages_custom(
        f,
        messages_area,
        app,
        user_style,
        assistant_style,
        highlight_style,
    );

    let status_text = app.status_message.as_deref().unwrap_or("");
    let status_bar = Paragraph::new(status_text)
        .style(if app.is_loading {
            loading_style
        } else {
            status_style
        })
        .alignment(Alignment::Left);
    f.render_widget(status_bar, chunks[1]);

    // Prepare conversation title outside of the input rendering
    let conv_title = app.cached_conversation_title.clone().unwrap_or_else(|| {
        let title = app
            .current_conversation_id
            .as_ref()
            .and_then(|id| {
                app.db_conn
                    .query_row(
                        "SELECT title FROM conversations WHERE id = ?1",
                        params![id],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()
                    .ok()
                    .flatten()
            })
            .map(|t| {
                if t.len() > 20 {
                    format!("{}...", t.chars().take(17).collect::<String>())
                } else {
                    t
                }
            })
            .unwrap_or_else(|| "New Chat".to_string());

        // Cache the title
        app.cached_conversation_title = Some(title.clone());
        title
    });

    let input_title = format!(
        "To {} (ID: {}...)",
        conv_title,
        app.current_conversation_id
            .as_deref()
            .map(|s| s.chars().take(8).collect::<String>())
            .unwrap_or("N/A".into())
    );

    let input_paragraph = Paragraph::new(app.input.as_str())
        .style(if app.is_loading {
            status_style // Dim input when loading
        } else {
            base_style
        })
        .block(Block::default().borders(Borders::ALL).title(Span::styled(
            input_title, // Dynamic title for input
            input_block_title_style,
        )));
    f.render_widget(input_paragraph, chunks[2]);

    // Only show cursor in input field if in Chatting mode and not loading and no external editor
    if !app.is_loading
        && matches!(app.app_mode, AppMode::Chatting)
        && !app.is_external_editor_active
    {
        f.set_cursor_position(Position::new(
            chunks[2].x + app.input_cursor_position as u16 + 1, // +1 for border
            chunks[2].y + 1,                                    // +1 for border
        ));
    }
}

fn render_messages_custom(
    f: &mut Frame,
    area: Rect,
    app: &mut App,
    user_style: Style,
    assistant_style: Style,
    highlight_style: Style,
) {
    if app.messages.is_empty() {
        return;
    }

    let highlight_symbol_len = HIGHLIGHT_SYMBOL.len() as u16;
    let selected_message = app.selected_message; // Extract before mutable borrow
    let available_height = area.height as usize;

    // Heuristic for message block height: avoid tiny scrollable areas if pane is small.
    // If pane is very small, allow messages to take more than 1/3rd.
    let max_message_block_height = if available_height < 10 {
        (available_height / 2).max(1) // If very small pane, allow up to half per message block
    } else {
        (available_height / 3).max(1) // Otherwise, 1/3rd is reasonable
    };

    // Pre-compute indentation strings for all message roles
    let user_prefix = "You: ";
    let assistant_prefix = "AI:  "; // Extra space for alignment
    let system_prefix = "Sys: ";
    let user_indent = " ".repeat(user_prefix.len());
    let assistant_indent = " ".repeat(assistant_prefix.len());
    let system_indent = " ".repeat(system_prefix.len());

    // Calculate wrapped lines for all messages first
    let mut message_display_data: Vec<(Vec<Line>, bool, usize)> = Vec::new(); // Added start_line_idx
    let mut total_lines = 0;

    for (msg_idx, msg) in app.messages.iter_mut().enumerate() {
        let style = if msg.role == Role::User {
            user_style
        } else {
            assistant_style
        };

        let (prefix_str, indentation) = match msg.role {
            Role::User => (user_prefix, &user_indent),
            Role::Assistant => (assistant_prefix, &assistant_indent),
            _ => (system_prefix, &system_indent),
        };

        let prefix_len = prefix_str.len() as u16;

        // Ensure content_wrap_width is at least 1
        let content_wrap_width = area
            .width
            .saturating_sub(prefix_len) // Space for prefix
            .saturating_sub(highlight_symbol_len.saturating_add(1)) // Space for highlight symbol and a margin
            .max(1);

        // Only recalculate wrapped lines if the width changed or not calculated yet
        let wrapped_strings = if msg.cached_wrap_width != Some(content_wrap_width)
            || msg.cached_wrapped_lines.is_none()
        {
            let wrapped = textwrap::wrap(&msg.content, content_wrap_width as usize);
            // Convert Cow<'_, str> to String for caching
            let wrapped_strings: Vec<String> =
                wrapped.into_iter().map(|cow| cow.into_owned()).collect();
            msg.cached_wrapped_lines = Some(wrapped_strings.clone());
            msg.cached_wrap_width = Some(content_wrap_width);
            wrapped_strings
        } else {
            msg.cached_wrapped_lines.clone().unwrap()
        };

        let mut all_lines: Vec<Line> = vec![];

        let prefix_span = Span::styled(prefix_str, style.add_modifier(Modifier::BOLD));

        if let Some(first_wrapped_line) = wrapped_strings.first() {
            all_lines.push(Line::from(vec![
                prefix_span.clone(),
                Span::styled(first_wrapped_line.clone(), style),
            ]));
        } else if !msg.content.is_empty() {
            // Handle case where content is not empty but wrapping results in no lines
            all_lines.push(Line::from(vec![
                prefix_span.clone(),
                Span::styled(msg.content.clone(), style), // Show original content, may overflow
            ]));
        } else {
            // Empty content (e.g. AI is typing but sent no content yet)
            all_lines.push(Line::from(prefix_span.clone()));
        }

        for line_content in wrapped_strings.iter().skip(1) {
            all_lines.push(Line::from(vec![
                Span::raw(indentation.clone()),
                Span::styled(line_content.clone(), style),
            ]));
        }

        // Apply max message height limit and auto-scroll to show newest text
        let mut display_lines = if all_lines.len() > max_message_block_height {
            let start_idx = all_lines.len().saturating_sub(max_message_block_height);
            all_lines[start_idx..].to_vec()
        } else {
            all_lines
        };

        // Add top border line for each message (except the first one) AFTER height limiting
        if msg_idx > 0 {
            let border_width = area.width.saturating_sub(highlight_symbol_len + 1) as usize;
            let border_line = "─".repeat(border_width);
            let border = Line::from(vec![Span::styled(
                border_line,
                Style::default().fg(Color::DarkGray),
            )]);
            // Insert at the beginning so border is always visible
            display_lines.insert(0, border);
        }

        let is_selected = selected_message == Some(msg_idx);
        let message_start_line = total_lines;
        total_lines += display_lines.len();
        message_display_data.push((display_lines, is_selected, message_start_line));
    }

    // Calculate scroll position to ensure selected message is visible
    let mut start_line = app.scroll_offset;

    // If we have a selected message, ensure it's visible
    if let Some(selected_idx) = selected_message {
        if selected_idx < message_display_data.len() {
            let (_, _, selected_start_line) = &message_display_data[selected_idx];
            let selected_end_line = if selected_idx + 1 < message_display_data.len() {
                message_display_data[selected_idx + 1].2
            } else {
                total_lines
            };

            // If selected message is above the viewport, scroll up
            if *selected_start_line < start_line {
                start_line = *selected_start_line;
            }
            // If selected message is below the viewport, scroll down
            else if selected_end_line > start_line + available_height {
                start_line = selected_end_line.saturating_sub(available_height);
            }
        }
    }
    // If no selection, auto-scroll to bottom to show latest messages
    else if total_lines > available_height {
        start_line = total_lines.saturating_sub(available_height);
    }

    // Update app's scroll offset
    app.scroll_offset = start_line;

    // Render messages line by line
    let mut current_y = area.y;
    let mut current_line = 0;

    for (_msg_idx, (lines, is_selected, _)) in message_display_data.iter().enumerate() {
        for (line_idx, line) in lines.iter().enumerate() {
            if current_line >= start_line && current_y < area.y + area.height {
                // Check if this is a border line (first line of messages after the first message)
                let is_border_line =
                    line.spans.len() == 1 && line.spans[0].content.chars().all(|c| c == '─');

                // Determine if this line should be highlighted
                let line_style = if *is_selected && !is_border_line {
                    highlight_style
                } else {
                    Style::default()
                };

                // Create the line content with highlight symbol if needed
                let display_line = if is_border_line {
                    // Border lines don't get highlight symbols, just render as-is with padding
                    let mut spans = vec![Span::raw(" ".repeat(highlight_symbol_len as usize))];
                    spans.extend_from_slice(&line.spans);
                    Line::from(spans)
                } else if *is_selected && line_idx == 0 {
                    // Add highlight symbol to first line of selected message
                    let mut spans = vec![Span::styled(HIGHLIGHT_SYMBOL, line_style)];
                    spans.extend_from_slice(&line.spans);
                    Line::from(spans)
                } else if *is_selected {
                    // Add spaces to align with highlight symbol for continuation lines
                    let mut spans = vec![Span::raw(" ".repeat(highlight_symbol_len as usize))];
                    spans.extend_from_slice(&line.spans);
                    Line::from(spans)
                } else {
                    // No selection, just add spaces where highlight symbol would be
                    let mut spans = vec![Span::raw(" ".repeat(highlight_symbol_len as usize))];
                    spans.extend_from_slice(&line.spans);
                    Line::from(spans)
                };

                // Render the line
                let line_area = Rect {
                    x: area.x,
                    y: current_y,
                    width: area.width,
                    height: 1,
                };

                let paragraph = Paragraph::new(Text::from(vec![display_line])).style(line_style);
                f.render_widget(paragraph, line_area);

                current_y += 1;
            }
            current_line += 1;
        }
    }
}

fn ui_picker(f: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let popup_area = centered_rect(80, 85, f.area()); // 80% width, 85% height

    f.render_widget(Clear, popup_area); // Clear the area for the popup

    let list_items: Vec<ListItem> = app
        .picker_items // picker_items is the filtered list
        .iter()
        .map(|item| {
            // Calculate the available width for content (accounting for borders and highlight symbol)
            let highlight_width = HIGHLIGHT_SYMBOL.chars().count() as u16;
            let available_width = popup_area
                .width
                .saturating_sub(2) // Subtract 2 for borders
                .saturating_sub(highlight_width + 1); // Highlight symbol + 1 space buffer

            // Fixed date format and length
            let date_text = item.updated_at.format(" (%y-%m-%d %H:%M)").to_string();
            let date_len = date_text.chars().count() as u16;

            // Always reserve space for the date plus a small buffer
            let date_space = date_len + 2; // 2 chars buffer

            // Calculate how much space is left for the title
            let max_title_len = available_width.saturating_sub(date_space);

            // Truncate title if needed
            let title_text = if item.title.chars().count() as u16 > max_title_len {
                let mut shortened = item
                    .title
                    .chars()
                    .take(max_title_len as usize - 3)
                    .collect::<String>();
                shortened.push_str("...");
                shortened
            } else {
                item.title.clone()
            };

            // Get the actual title length before styling
            let actual_title_len = title_text.chars().count() as u16;

            // Style the title and date (clone title_text to avoid move)
            let title = Span::styled(title_text.clone(), theme.base_style);

            // Create a line with the title left-aligned and date right-aligned
            let mut line = Line::default();
            line.spans.push(title);

            // Calculate padding to push date to the right
            let padding = available_width
                .saturating_sub(actual_title_len)
                .saturating_sub(date_len);
            line.spans
                .push(Span::styled(" ".repeat(padding as usize), Style::default()));

            // Add the date with dark gray color
            line.spans.push(Span::styled(
                date_text,
                theme.status_style.fg(Color::DarkGray),
            ));

            ListItem::new(line)
        })
        .collect();

    // Render the list widget without any title
    let list_widget = List::new(list_items)
        .block(Block::default().borders(Borders::ALL))
        .highlight_style(theme.highlight_style.clone())
        .highlight_symbol(HIGHLIGHT_SYMBOL);

    f.render_stateful_widget(list_widget, popup_area, &mut app.picker_state);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ]
            .as_ref(),
        )
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ]
            .as_ref(),
        )
        .split(popup_layout[1])[1]
}
