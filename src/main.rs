use async_openai::{
    Client,
    config::AzureConfig,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, Role,
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
    openai_endpoints: Option<HashMap<String, OpenAIEndpointToml>>,
}

#[derive(Deserialize, Debug, Clone)]
struct OpenAIEndpointToml {
    api_key: Option<String>,
    api_base: Option<String>,
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

// --- Application State and Messages ---
#[derive(Clone)]
struct AppMessage {
    role: Role,
    content: String,
    timestamp: DateTime<Utc>,
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

// Create an enum to handle different config types
enum ClientConfig {
    OpenAI(OpenAIConfig),
    Azure(AzureConfig),
}

struct App<'a> {
    input: String,
    messages: Vec<AppMessage>,
    openai_client: Client<OpenAIConfig>,
    model_to_use: String,
    is_loading: bool,
    _config_source_message: String,
    update_sender: mpsc::Sender<AppUpdate>,
    update_receiver: mpsc::Receiver<AppUpdate>,
    message_list_state: ListState,
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
        // Catppuccin Macchiato Colors
        let base_bg = Color::Rgb(0x1e, 0x1e, 0x2e); // Base
        let text_fg = Color::Rgb(0xca, 0xd3, 0xf5); // Text

        let user_fg = Color::Rgb(0x89, 0xdc, 0xeb); // Sky (was Cyan)
        let assistant_fg = Color::Rgb(0xa6, 0xe3, 0xa1); // Green

        let title_fg = Color::Rgb(0xcb, 0xa6, 0xf7); // Mauve (for general titles)

        let loading_fg = Color::Rgb(0xfa, 0xb3, 0x87); // Peach (was Magenta)
        let status_fg = Color::Rgb(0xa6, 0xad, 0xc8); // Subtext0 (was Gray)

        let highlight_fg = Color::Rgb(0xb4, 0xbe, 0xfe); // Lavender
        let highlight_bg = Color::Rgb(0x45, 0x47, 0x5a); // Surface1 (was LightMagenta BG)

        let picker_title_fg = Color::Rgb(0xf5, 0xc2, 0xe7); // Pink (was Magenta)

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
            picker_title_style: Style::default()
                .fg(picker_title_fg)
                .add_modifier(Modifier::BOLD),
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
        client: Client<OpenAIConfig>,
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
            message_list_state: ListState::default(),
            input_cursor_position: 0,
            status_message: Some(
                "Ctrl+O:Open Ctrl+E:EditMsg Ctrl+T:EditInput Ctrl+X:Cancel Enter:Send Ctrl+C:Quit."
                    .to_string(),
            ),
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
        });
        if let Err(e) =
            self.add_message_to_db(&current_conv_id, Role::User, &user_message_content, now)
        {
            self.status_message = Some(format!("Error saving message: {}", e));
        }
        self.message_list_state
            .select(Some(self.messages.len().saturating_sub(1)));
        self.input.clear();
        self.input_cursor_position = 0;
        self.is_loading = true;
        self.status_message = Some("Sending to AI...".to_string());
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

            let stream_result = client.chat().create_stream(request_args.unwrap()).await;

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

        let old_selected_id: Option<String> = self
            .picker_state
            .selected()
            .and_then(|idx| self.picker_items.get(idx).map(|item| item.id.clone()));

        self.picker_items.clear();

        self.picker_items.push(ConversationMeta {
            id: "NEW_CHAT".to_string(),
            title: "[+] New Chat".to_string(),
            updated_at: Utc::now(),
        });

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

        if let Some(id_to_reselect) = old_selected_id {
            if let Some(new_idx) = self
                .picker_items
                .iter()
                .position(|item| item.id == id_to_reselect)
            {
                self.picker_state.select(Some(new_idx));
            } else if !self.picker_items.is_empty() {
                self.picker_state.select(Some(0));
            } else {
                self.picker_state.select(None);
            }
        } else if !self.picker_items.is_empty() {
            self.picker_state.select(Some(0));
        } else {
            self.picker_state.select(None);
        }

        if let Some(selected_idx) = self.picker_state.selected() {
            if selected_idx >= self.picker_items.len() && !self.picker_items.is_empty() {
                self.picker_state.select(Some(self.picker_items.len() - 1));
            } else if self.picker_items.is_empty() {
                self.picker_state.select(None);
            }
        } else if !self.picker_items.is_empty() {
            self.picker_state.select(Some(0));
        }
    }

    fn handle_chatting_input(
        &mut self,
        key_code: KeyCode,
        key_modifiers: KeyModifiers,
    ) -> AppAction {
        if key_modifiers == KeyModifiers::CONTROL {
            match key_code {
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
                        let content_to_edit = if let Some(idx) = self.message_list_state.selected()
                        {
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
                    match self.message_list_state.selected() {
                        Some(idx) if idx > 0 => self.message_list_state.select(Some(idx - 1)),
                        Some(_) => self.message_list_state.select(Some(0)),
                        None => self
                            .message_list_state
                            .select(Some(self.messages.len() - 1)),
                    }
                }
            }
            KeyCode::Down => {
                if !self.messages.is_empty() {
                    match self.message_list_state.selected() {
                        Some(idx) if idx < self.messages.len() - 1 => {
                            self.message_list_state.select(Some(idx + 1))
                        }
                        Some(idx) => self.message_list_state.select(Some(idx)),
                        None => self.message_list_state.select(Some(0)),
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
            KeyCode::Up => {
                if !self.picker_items.is_empty() {
                    let current = self.picker_state.selected().unwrap_or(0);
                    self.picker_state.select(Some(current.saturating_sub(1)));
                }
            }
            KeyCode::Down => {
                if !self.picker_items.is_empty() {
                    let current = self.picker_state.selected().unwrap_or(0);
                    if current < self.picker_items.len() - 1 {
                        self.picker_state.select(Some(current + 1));
                    }
                }
            }
            KeyCode::Enter => {
                if let Some(idx) = self.picker_state.selected() {
                    if idx < self.picker_items.len() {
                        let item = self.picker_items[idx].clone();
                        if item.id == "NEW_CHAT" {
                            self.messages.clear();
                            self.current_conversation_id = None;
                            self.input.clear();
                            self.message_list_state.select(None);
                            self.status_message = Some("New chat started.".to_string());
                        } else {
                            if let Err(e) = self.load_selected_conversation(&item.id) {
                                self.status_message = Some(format!("Err load chat: {}", e));
                            } else {
                                self.status_message = Some(format!("Loaded: {}", item.title));
                            }
                        }
                        self.app_mode = AppMode::Chatting;
                        self.picker_filter_input.clear();
                    }
                }
            }
            KeyCode::Char(c) if c == 'x' && key_modifiers == KeyModifiers::CONTROL => {
                if let Some(selected_idx) = self.picker_state.selected() {
                    if selected_idx < self.picker_items.len() {
                        let item_to_delete = self.picker_items[selected_idx].clone();

                        if item_to_delete.id == "NEW_CHAT" {
                            self.status_message = Some("Cannot delete '[+] New Chat'.".to_string());
                            return;
                        }

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
        self.message_list_state.select(if self.messages.is_empty() {
            None
        } else {
            Some(self.messages.len() - 1)
        });
        Ok(())
    }

    fn update_from_channel(&mut self) {
        match self.update_receiver.try_recv() {
            Ok(AppUpdate::AssistantChunk(chunk)) => {
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
                        });
                        is_new_msg = true;
                    }
                } else if let Some(last) = self.messages.last_mut() {
                    if last.role == Role::Assistant {
                        last.content.push_str(&chunk);
                    } else if !chunk.is_empty() {
                        self.messages.push(AppMessage {
                            role: Role::Assistant,
                            content: chunk.clone(),
                            timestamp: Utc::now(),
                        });
                        is_new_msg = true;
                    }
                }

                if !self.is_external_editor_active {
                    self.status_message = Some("AI Typing...".to_string());
                    // Always keep the latest message selected when AI is typing
                    if !self.messages.is_empty() {
                        self.message_list_state
                            .select(Some(self.messages.len() - 1));
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
                self.is_loading = false;
                let content = format!("[Error]: {}", err_msg);
                let now = Utc::now();
                self.messages.push(AppMessage {
                    role: Role::Assistant,
                    content: content.clone(),
                    timestamp: now,
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
                        self.message_list_state
                            .select(Some(self.messages.len() - 1));
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
                                    self.status_message = Some(format!("Err save AI resp: {}", e));
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
                        self.message_list_state
                            .select(Some(self.messages.len().saturating_sub(1)));
                    } else {
                        self.message_list_state.select(None);
                    }
                }
                // If editor was active, user will close it. `run_app` handles process exit.
            }
            Ok(AppUpdate::AssistantCancelled) => {
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
                        self.message_list_state.select(None);
                    } else {
                        self.message_list_state
                            .select(Some(self.messages.len().saturating_sub(1)));
                    }
                } else {
                    eprintln!("{}", final_status_message);
                    // If editor was active, user will close it. `run_app` handles process exit.
                }
            }
            Err(mpsc::error::TryRecvError::Empty) => {}
            Err(mpsc::error::TryRecvError::Disconnected) => {
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
            }
        }
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
    let toml_path = Path::new("endpoints.toml");
    let toml_config = load_toml_config(toml_path).unwrap_or_else(|err| {
        eprintln!(
            "Warning: Could not load '{}': {}. Using defaults.",
            toml_path.display(),
            err
        );
        EndpointsTomlConfig::default()
    });
    let mut selected_endpoint_details: Option<OpenAIEndpointToml> = None;
    let mut config_source_message: String = "Defaults (e.g. OPENAI_API_KEY env var)".to_string();
    let endpoint_name_to_load: Option<String> = chosen_endpoint_name_from_cli
        .clone()
        .or_else(|| toml_config.default_openai_endpoint.clone());
    if let Some(name) = &endpoint_name_to_load {
        if let Some(endpoints_map) = &toml_config.openai_endpoints {
            if let Some(details) = endpoints_map.get(name) {
                selected_endpoint_details = Some(details.clone());
                config_source_message = format!("endpoint '{}' from TOML", name);
            } else {
                eprintln!("Warning: Endpoint '{}' not found in TOML.", name);
            }
        } else if chosen_endpoint_name_from_cli.is_some()
            || toml_config.default_openai_endpoint.is_some()
        {
            eprintln!(
                "Warning: Endpoint '{}' specified, but no [openai_endpoints] table in TOML.",
                name
            );
        }
    } else if let Some(endpoints_map) = &toml_config.openai_endpoints {
        if let Some((name, details)) = endpoints_map.iter().next() {
            selected_endpoint_details = Some(details.clone());
            config_source_message = format!("first available endpoint '{}' from TOML", name);
        }
    }
    let mut openai_client_config = OpenAIConfig::new();
    let mut model_to_use = "gpt-4o".to_string(); // Default model
    if let Some(details) = &selected_endpoint_details {
        if let Some(key) = &details.api_key {
            openai_client_config = openai_client_config.with_api_key(key);
        }
        if let Some(base) = &details.api_base {
            openai_client_config = openai_client_config.with_api_base(base);
        }
        if let Some(model) = &details.default_model {
            model_to_use = model.clone();
        }
    }
    let client = Client::with_config(openai_client_config);
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
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = StdInstant::now();
    let mut content_for_blocking_editor: Option<String> = None;

    loop {
        app.update_from_channel(); // Always process incoming AI updates, which might update editor_file

        // --- Streaming Editor Management ---
        if app.editor_process.is_some() {
            // Actively manage if a process was spawned
            let mut editor_finished_this_tick = false;
            if let Some(editor_process_mut) = &mut app.editor_process {
                match editor_process_mut.try_wait() {
                    Ok(Some(_exit_status)) => {
                        // Editor exited
                        editor_finished_this_tick = true;
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
                    }
                }
            }

            if editor_finished_this_tick {
                app.editor_process = None; // Clear the process
                app.editor_file = None; // Drop NamedTempFile, which deletes the file
                app.is_external_editor_active = false;

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
            }
        } else {
            // If editor_process is None, ensure is_external_editor_active is false.
            // This handles cases where the process might have been cleared by other logic (e.g. error in update_from_channel)
            if app.is_external_editor_active {
                app.is_external_editor_active = false;
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
            }
        }

        // If an external editor is determined to be active (e.g. editor_process.try_wait() returned None),
        // we skip the rest of the TUI loop for this iteration.
        if app.is_external_editor_active {
            continue;
        }

        // --- Standard TUI Processing (only if no external editor is active) ---
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c') {
                        if let Some(mut child) = app.editor_process.take() {
                            // if streaming editor is running
                            let _ = child.kill(); // Attempt to kill it
                            let _ = child.wait(); // Wait for it
                        }
                        app.editor_file = None; // Ensure temp file is dropped/deleted
                        return Ok(()); // Exit app
                    }

                    if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('o') {
                        app.app_mode = AppMode::PickingConversation;
                        app.picker_filter_input.clear();
                        if let Err(e) = app.load_all_conversations_from_db() {
                            app.status_message =
                                Some(format!("Error loading conversations: {}", e));
                            app.app_mode = AppMode::Chatting;
                        } else {
                            app.apply_filter_and_update_picker_items();
                        }
                        terminal.draw(|f| ui(f, app))?;
                        last_tick = StdInstant::now();
                        continue;
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
                            // terminal.clear() might not be needed if editor clears, but can't hurt.
                            // No, terminal.clear() uses TUI clear, which is bad here.
                            // The editor should handle its own screen.

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
                                            // For Windows, if editor is GUI (notepad), it detaches.
                                            // If it's a console editor like vim.exe, it takes over.
                                            // Using `start /B` could run console apps without new window but still in background.
                                            // For simplicity, direct spawn:
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
                                                // Don't set TUI status_message, it's not visible.
                                                // eprintln used for out-of-band info if needed.
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
                                }
                            }
                        }
                        AppAction::None => {}
                    }
                }
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
            // Don't call terminal.clear() here before external editor

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
            // Force redraw TUI
            terminal.draw(|f| ui(f, app))?;
            last_tick = StdInstant::now();
        } else if !app.is_external_editor_active {
            // Only draw if no editor is active and not just handled blocking editor
            if last_tick.elapsed() >= tick_rate || timeout == Duration::from_secs(0) {
                terminal.draw(|f| ui(f, app))?;
                last_tick = StdInstant::now();
            }
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

    let chat_area_width = chunks[0].width;
    let chat_pane_height = chunks[0].height;
    let highlight_symbol_len = HIGHLIGHT_SYMBOL.len() as u16;

    // Heuristic for message block height: avoid tiny scrollable areas if pane is small.
    // If pane is very small, allow messages to take more than 1/3rd.
    let max_message_block_height = if chat_pane_height < 10 {
        (chat_pane_height / 2).max(1) // If very small pane, allow up to half per message block
    } else {
        (chat_pane_height / 3).max(1) // Otherwise, 1/3rd is reasonable
    };

    let display_messages: Vec<ListItem> = app
        .messages
        .iter()
        .map(|msg| {
            let style = if msg.role == Role::User {
                theme.user_style
            } else {
                theme.assistant_style
            };
            let prefix_str = match msg.role {
                Role::User => "You: ",
                Role::Assistant => "AI:  ", // Extra space for alignment
                _ => "Sys: ",
            };
            let prefix_len = prefix_str.len() as u16;
            let indentation = " ".repeat(prefix_str.len());

            // Ensure content_wrap_width is at least 1
            let content_wrap_width = chat_area_width
                .saturating_sub(prefix_len) // Space for prefix
                .saturating_sub(highlight_symbol_len.saturating_add(1)) // Space for highlight symbol and a margin
                .max(1);

            let wrapped_strings = textwrap::wrap(&msg.content, content_wrap_width as usize);

            let mut all_lines: Vec<Line> = vec![];
            let prefix_span = Span::styled(prefix_str, style.add_modifier(Modifier::BOLD));

            if let Some(first_wrapped_line) = wrapped_strings.first() {
                all_lines.push(Line::from(vec![
                    prefix_span.clone(),
                    Span::styled(first_wrapped_line.clone(), style),
                ]));
            } else if !msg.content.is_empty() {
                // Handle case where content is not empty but wrapping results in no lines (e.g. too small width)
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

            // If content is larger than max height, show the last lines (auto-scroll)
            let display_lines = if all_lines.len() > max_message_block_height as usize {
                let start_idx = all_lines
                    .len()
                    .saturating_sub(max_message_block_height as usize);
                all_lines[start_idx..].to_vec()
            } else {
                all_lines
            };

            ListItem::new(Text::from(display_lines))
        })
        .collect();

    let messages_list = List::new(display_messages)
        .block(
            Block::default()
                .borders(Borders::NONE) // No border for message pane itself
                .title(Span::styled(
                    APP_TITLE,
                    theme.chat_block_title_style.clone(),
                ))
                .title_alignment(Alignment::Center),
        )
        .highlight_style(theme.highlight_style.clone())
        .highlight_symbol(HIGHLIGHT_SYMBOL);

    f.render_stateful_widget(messages_list, chunks[0], &mut app.message_list_state);

    let status_text = app.status_message.as_deref().unwrap_or(
        "Ctrl+O:Open Ctrl+E:EditMsg Ctrl+T:EditInput Ctrl+X:Cancel Enter:Send Ctrl+C:Quit.",
    );
    let status_bar = Paragraph::new(status_text)
        .style(if app.is_loading {
            theme.loading_style
        } else {
            theme.status_style
        })
        .alignment(Alignment::Left);
    f.render_widget(status_bar, chunks[1]);

    let conv_title = app
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
            theme.status_style // Dim input when loading
        } else {
            theme.base_style
        })
        .block(Block::default().borders(Borders::ALL).title(Span::styled(
            input_title, // Dynamic title for input
            theme.input_block_title_style.clone(),
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

fn ui_picker(f: &mut Frame, app: &mut App) {
    let theme = &app.theme;
    let popup_area = centered_rect(80, 85, f.area()); // 80% width, 85% height

    f.render_widget(Clear, popup_area); // Clear the area for the popup

    let list_items: Vec<ListItem> = app
        .picker_items // picker_items is the filtered list
        .iter()
        .map(|item| {
            let title_style = if item.id == "NEW_CHAT" {
                theme.base_style.fg(Color::LightMagenta) // Style "New Chat" differently
            } else {
                theme.base_style
            };
            let title = Span::styled(item.title.clone(), title_style);

            if item.id == "NEW_CHAT" {
                ListItem::new(Line::from(vec![title]))
            } else {
                let date = Span::styled(
                    item.updated_at.format(" (%y-%m-%d %H:%M)").to_string(),
                    theme.status_style.fg(Color::DarkGray), // A less prominent color for date
                );
                ListItem::new(Line::from(vec![title, date]))
            }
        })
        .collect();

    let filter_display_text = if app.picker_filter_input.is_empty() {
        "Type to filter".to_string()
    } else {
        format!("Filter: {}", app.picker_filter_input)
    };

    let picker_block_title = format!(
        "{} | Open/New (Esc:Close Ctrl+X:Delete)",
        filter_display_text
    );

    let list_widget = List::new(list_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    picker_block_title,
                    theme.picker_title_style.clone(),
                ))
                .title_alignment(Alignment::Center),
        )
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
