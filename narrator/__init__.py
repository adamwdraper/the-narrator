"""
The Narrator - Thread and file storage components for conversational AI
"""

from .database.thread_store import ThreadStore
from .storage.file_store import FileStore
from .models.thread import Thread
from .models.message import Message
from .models.attachment import Attachment
from .utils.registry import (
    register_thread_store,
    get_thread_store,
    register_file_store,
    get_file_store,
    register,
    get,
    list as list_registered_components,
)

__version__ = "0.1.0"
__all__ = [
    "ThreadStore",
    "FileStore", 
    "Thread",
    "Message",
    "Attachment",
    "register_thread_store",
    "get_thread_store",
    "register_file_store",
    "get_file_store",
    "register",
    "get",
    "list_registered_components",
] 