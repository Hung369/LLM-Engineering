import logging
import os
import tempfile
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler
from telegram.ext import filters, ContextTypes, ConversationHandler
from dotenv import load_dotenv
from Stark import LlamaBot

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.environ.get('BOT_TOKEN')
ADMIN_USER_IDS = os.environ.get('ADMIN_USER_IDS', '').split(',')

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the bot with hierarchical PDF RAG capabilities
SYSTEM_PROMPT = """You are Stark, an AI assistant specialized in answering questions about PDF documents.
You respond only in English with clear, concise replies and always cite the specific parts of documents 
when providing information.

The following context comes from PDF documents:

{context}

When answering, mention the document name, page number, and section when relevant."""

llama_bot = LlamaBot( max_history=10, system_prompt=SYSTEM_PROMPT )

# Conversation states
WAITING_FOR_DOC_TYPE, WAITING_FOR_PDF_UPLOAD = range(2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user_id = str(update.effective_user.id)
    user_name = update.effective_user.first_name
    
    # Make sure the user has a chat history initialized
    llama_bot.get_history(user_id)
    
    await update.message.reply_text(
        f"*Welcome to Stark PDF Assistant, {user_name}!*\n\n"
        "I'm Stark, your AI assistant specialized in answering questions about PDF documents.\n\n"
        "📚 I can help you extract insights from PDF files using hierarchical retrieval.\n\n"
        "_Use /help to see available commands._",
        parse_mode=ParseMode.MARKDOWN
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    user_id = str(update.effective_user.id)
    is_admin = user_id in ADMIN_USER_IDS
    
    basic_commands = (
        "*Available Commands:*\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/reset - Reset your chat history\n"
        "/documents - List available PDF documents\n"
    )
    
    admin_commands = ""
    if is_admin:
        admin_commands = (
            "\n*Admin Commands:*\n"
            "/addpdf - Upload a new PDF document\n"
            "/structure - View the hierarchical structure of a document\n"
        )
    
    usage_info = (
        "\n*How to use:*\n"
        "1. Upload PDFs using /addpdf (admin only)\n"
        "2. Ask questions about the documents\n"
        "3. I'll retrieve relevant sections using hierarchical RAG\n"
        "4. My answers will include document, page, and section references\n\n"
        "_For best results, ask specific questions about the content in the PDFs._"
    )
    
    await update.message.reply_text(
        basic_commands + admin_commands + usage_info,
        parse_mode=ParseMode.MARKDOWN
    )

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset the chat history when the command /reset is issued."""
    user_id = str(update.effective_user.id)
    llama_bot.reset_history(user_id)
    await update.message.reply_text(
        "*Chat history has been reset!*\n\n"
        "_Starting fresh with default context._",
        parse_mode=ParseMode.MARKDOWN
    )

async def list_documents(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all PDF documents in the knowledge base with hierarchy info."""
    try:
        docs = await llama_bot.list_documents()
        
        if not docs:
            await update.message.reply_text(
                "*No PDF documents found in the knowledge base.*\n\n"
                "_Add documents using the /addpdf command (admin only)._",
                parse_mode=ParseMode.MARKDOWN
            )
            return
            
        message = "*Available PDF Documents:*\n\n"
        for i, doc in enumerate(docs, 1):
            message += f"{i}. *{doc['filename']}*\n"
            message += f"   Pages: {doc['total_pages']}\n"
            message += f"   Sections: {doc['total_sections']}\n"
            message += f"   Chunks: {doc['total_chunks']}\n"
            
            # Add button to view structure (for admins)
            if str(update.effective_user.id) in ADMIN_USER_IDS:
                message += f"   _Use /structure {doc['doc_id']} to view detailed structure_\n"
            
            message += "\n"
            
        await update.message.reply_text(
            message,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        await update.message.reply_text(
            f"*Error listing documents:*\n`{str(e)}`",
            parse_mode=ParseMode.MARKDOWN
        )

async def view_document_structure(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View the hierarchical structure of a specific document."""
    user_id = str(update.effective_user.id)
    
    # Check if user is an admin
    if user_id not in ADMIN_USER_IDS:
        await update.message.reply_text(
            "*Permission denied.*\n\n"
            "_Only admins can view document structures._",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Get document ID from command args
    if not context.args or len(context.args) < 1:
        await update.message.reply_text(
            "*Please provide a document ID.*\n\n"
            "_Example: /structure doc_123_",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    doc_id = context.args[0]
    
    try:
        structure = await llama_bot.get_document_structure(doc_id)
        
        if "error" in structure:
            await update.message.reply_text(
                f"*Error getting document structure:*\n`{structure['error']}`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        # Build a hierarchical display of the document
        message = f"*Document Structure: {structure.get('filename', 'Unknown')}*\n\n"
        message += f"Document ID: `{doc_id}`\n"
        message += f"Total Pages: {structure.get('total_pages', 0)}\n\n"
        
        # Show page and section information
        pages = structure.get("sections", {})
        if pages:
            for page_num, sections in sorted(pages.items(), key=lambda x: int(x[0])):
                message += f"📄 *Page {page_num}* ({len(sections)} sections)\n"
                
                # Limit to first 3 sections per page to avoid message size issues
                for i, section in enumerate(sections[:3]):
                    section_num = section.get("section_num", i+1)
                    content_preview = section.get("content_preview", "")[:50]
                    chunks = len(section.get("chunks", []))
                    
                    message += f"  📌 Section {section_num}: {content_preview}... ({chunks} chunks)\n"
                
                if len(sections) > 3:
                    message += f"  _...and {len(sections) - 3} more sections_\n"
                
                message += "\n"
        
        await update.message.reply_text(
            message,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Error viewing document structure: {e}")
        await update.message.reply_text(
            f"*Error viewing document structure:*\n`{str(e)}`",
            parse_mode=ParseMode.MARKDOWN
        )

async def add_pdf_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the process of adding a new PDF document."""
    user_id = str(update.effective_user.id)
    
    # Check if user is an admin
    if user_id not in ADMIN_USER_IDS:
        await update.message.reply_text(
            "*Permission denied.*\n\n"
            "_Only admins can add PDF documents to the knowledge base._",
            parse_mode=ParseMode.MARKDOWN
        )
        return ConversationHandler.END
    
    await update.message.reply_text(
        "*Adding a new PDF document to the knowledge base*\n\n"
        "Please enter a document type/category (e.g., 'manual', 'report', 'article'):\n\n"
        "_Send /cancel to abort this operation._",
        parse_mode=ParseMode.MARKDOWN
    )
    
    return WAITING_FOR_DOC_TYPE

async def add_pdf_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process the document type and ask for PDF upload."""
    doc_type = update.message.text.strip()
    
    # Save the doc_type in user_data for later use
    context.user_data['doc_type'] = doc_type
    
    await update.message.reply_text(
        f"*Document type set to:* `{doc_type}`\n\n"
        "Now, please upload the PDF file.\n\n"
        "_Send /cancel to abort this operation._",
        parse_mode=ParseMode.MARKDOWN
    )
    
    return WAITING_FOR_PDF_UPLOAD

async def handle_pdf_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process the uploaded PDF document and add it to the knowledge base."""
    doc_type = context.user_data.get('doc_type', 'general')
    
    # Check if a document was uploaded
    if not update.message.document:
        await update.message.reply_text(
            "*Please upload a PDF document.*\n\n"
            "_Send /cancel to abort this operation._",
            parse_mode=ParseMode.MARKDOWN
        )
        return WAITING_FOR_PDF_UPLOAD
    
    # Check if the document is a PDF
    document = update.message.document
    if not document.file_name.lower().endswith('.pdf'):
        await update.message.reply_text(
            "*Only PDF files are supported.*\n\n"
            "Please upload a file with .pdf extension.",
            parse_mode=ParseMode.MARKDOWN
        )
        return WAITING_FOR_PDF_UPLOAD
    
    # Let the user know we're processing
    processing_msg = await update.message.reply_text(
        f"_Downloading and processing {document.file_name}..._\n\n"
        "This may take some time depending on the PDF size and complexity.",
        parse_mode=ParseMode.MARKDOWN
    )
    
    try:
        # Download the file
        pdf_file = await context.bot.get_file(document.file_id)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = temp_file.name
        
        # Download to the temporary file
        await pdf_file.download_to_drive(temp_path)
        
        # Create destination path in data directory
        os.makedirs('pdf_data', exist_ok=True)
        destination_path = os.path.join('pdf_data', document.file_name)
        
        # Add the PDF to the knowledge base
        await processing_msg.edit_text(
            f"_PDF downloaded. Adding to knowledge base and creating embeddings..._\n\n"
            "This process will take longer for larger documents.",
            parse_mode=ParseMode.MARKDOWN
        )
        
        success = await llama_bot.add_pdf_document(temp_path, doc_type)
        
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {temp_path}: {e}")
        
        if success:
            await processing_msg.edit_text(
                f"*PDF document added successfully!*\n\n"
                f"Filename: {document.file_name}\n"
                f"Type: {doc_type}\n"
                f"Size: {document.file_size} bytes\n\n"
                "_The document has been processed, chunked hierarchically, and added to the knowledge base._",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await processing_msg.edit_text(
                "*Failed to add PDF document.*\n\n"
                "_There was an error processing the document. Please check the logs for more information._",
                parse_mode=ParseMode.MARKDOWN
            )
    except Exception as e:
        logger.error(f"Error adding PDF document: {e}")
        await processing_msg.edit_text(
            f"*Error adding PDF document:*\n`{str(e)}`",
            parse_mode=ParseMode.MARKDOWN
        )
    
    # Clear user_data
    context.user_data.clear()
    
    return ConversationHandler.END

async def cancel_operation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the current operation."""
    context.user_data.clear()
    
    await update.message.reply_text(
        "*Operation cancelled.*",
        parse_mode=ParseMode.MARKDOWN
    )
    
    return ConversationHandler.END

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process the user message and respond with hierarchical PDF RAG responses."""
    user_id = str(update.effective_user.id)
    user_message = update.message.text
    
    # Let the user know we're processing
    processing_msg = await update.message.reply_text(
        "_Processing your question using hierarchical PDF retrieval..._",
        parse_mode=ParseMode.MARKDOWN
    )
    
    try:
        # Get the response from LlamaBot with hierarchical PDF RAG
        response = await llama_bot.chat(user_id, user_message)
        
        # Attempt to use markdown for formatting, fallback to plain text if it fails
        try:
            await processing_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await processing_msg.edit_text(response)
        
    except Exception as e:
        logger.error(f"Error while processing message: {e}")
        await processing_msg.edit_text(
            f"*Sorry, an error occurred:*\n\n`{str(e)}`",
            parse_mode=ParseMode.MARKDOWN
        )

def main() -> None:
    """Start the bot."""
    # Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # PDF addition conversation handler
    pdf_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("addpdf", add_pdf_start)],
        states={
            WAITING_FOR_DOC_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, add_pdf_type)],
            WAITING_FOR_PDF_UPLOAD: [MessageHandler(filters.Document.ALL, handle_pdf_upload)],
        },
        fallbacks=[CommandHandler("cancel", cancel_operation)],
    )

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("documents", list_documents))
    application.add_handler(CommandHandler("structure", view_document_structure))
    application.add_handler(pdf_conv_handler)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))

    # Start the Bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()