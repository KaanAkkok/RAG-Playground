from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import Markdown, clear_output, display

from rag_playground.core.settings import PROJECT_ROOT
from rag_playground.ingestion.document_io import DocumentIO
from rag_playground.models.file_type_model import FileType


def create_upload_widget() -> widgets.FileUpload:
    """PDF dokümanı seçmek için dosya yükleme widget'ı oluşturur ve ekranda görüntüler.

    Girdi Parametreleri (Inputs):
        Parametre almaz.

    Döndürdüğü Değer (Output):
        widgets.FileUpload: PDF seçim widget'ı.

    Nasıl Kullanılır:
        >>> upload = create_upload_widget()
    """
    upload = widgets.FileUpload(
        accept=str(FileType.PDF),
        multiple=False,
        description="PDF Seç",
    )
    display(upload)
    return upload


async def save_uploaded_file(
    upload_widget: widgets.FileUpload,
    save_dir: Path | str | None = None,
    default_filename: str = "AS19_Sayi-RAG-Sistemlerine-Giris.pdf",
    encoding: str = "utf-8",
) -> Path:
    """Seçilen PDF'i diske kaydeder; dosya seçilmemişse hazır varsayılan PDF yolunu döner.

    Girdi Parametreleri (Inputs):
        upload_widget (widgets.FileUpload): create_upload_widget() ile oluşturulan widget.
        save_dir (Path | str | None): Dosyanın kaydedileceği hedef klasör.
            Varsayılan: PROJECT_ROOT / "data/documents/raw_documents"
        default_filename (str): Dosya seçilmediğinde kullanılacak varsayılan PDF dosya adı.
            Varsayılan: "AS19_Sayi-RAG-Sistemlerine-Giris.pdf"
        encoding (str): Karakter kodlaması (varsayılan: "utf-8").

    Döndürdüğü Değer (Output):
        Path: Kaydedilen veya varsayılan PDF dosyasının tam dosya yolu (saved_pdf_path).

    Nasıl Kullanılır:
        >>> upload = create_upload_widget()
        >>> saved_pdf_path = await save_uploaded_file(upload)
    """
    document_io = DocumentIO(encoding=encoding)
    raw_dir = Path(str(save_dir or (PROJECT_ROOT / "data/documents/raw_documents")))
    raw_dir.mkdir(parents=True, exist_ok=True)
    default_path = raw_dir / default_filename

    if upload_widget.value:
        # demo.ipynb dosyasındaki format:
        if isinstance(upload_widget.value, (list, tuple)):
            uploaded_file = upload_widget.value[0]
        else:
            uploaded_file = next(iter(upload_widget.value.values()))

        filename = Path(uploaded_file["name"]).stem
        content = bytes(uploaded_file["content"])
        saved_file_path = await document_io.save_document(
            content=content,
            filename=filename,
            save_dir=raw_dir,
            file_type=FileType.PDF,
        )
        print(f"Yeni dosya yüklendi ve kaydedildi: {saved_file_path}")
        return saved_file_path
    else:
        print(f"Dosya seçilmedi, varsayılan dosya kullanılıyor: {default_path}")
        return default_path


def create_chat_widget(chatbot: Any | None = None) -> None:
    """İnteraktif RAG konuşma arayüzünü (Chat Widget) oluşturur ve ekranda görüntüler.

    Girdi Parametreleri (Inputs):
        chatbot (AssistantService | None): Öğrencinin oluşturduğu asistan nesnesi.
            Eğer None verilirse, çalışma ortamındaki global 'chatbot' değişkenini kullanır.

    Nasıl Kullanılır:
        >>> create_chat_widget(chatbot=chatbot)
        # ya da chatbot yukarıda tanımlandıysa doğrudan:
        >>> create_chat_widget()
    """
    text_input = widgets.Text(
        value="",
        placeholder="Sorunuzu buraya yazın (örn: RAG mimarisi nedir?)...",
        description="Soru:",
        layout=widgets.Layout(width="70%"),
    )
    btn_send = widgets.Button(
        description="Gönder",
        button_style="primary",
        layout=widgets.Layout(width="14%"),
    )
    btn_clear = widgets.Button(
        description="Temizle",
        button_style="warning",
        layout=widgets.Layout(width="14%"),
    )
    chat_output = widgets.Output()

    def handle_query(b: Any = None) -> None:
        user_query = text_input.value.strip()
        if not user_query:
            return
        text_input.value = ""

        with chat_output:
            print(f"\nSoru: {user_query}")
            active_bot = chatbot or globals().get("chatbot")
            if active_bot is None:
                print(
                    "Hata: 'chatbot' nesnesi bulunamadı! Lütfen aşağıdaki çalışma alanında "
                    "AssistantService nesnenizi 'chatbot' adıyla oluşturup çalıştırın."
                )
                return

            async def run_ask() -> None:
                try:
                    response = await active_bot.answer(user_query)
                    display(Markdown(f"**Yanıt:**\n{response.answer}"))
                    if getattr(response, "citations", None):
                        print(
                            active_bot.show_citations(
                                citations=response.citations,
                                score_field="rerank_score",
                            )
                        )
                except Exception as e:
                    print(f"Hata oluştu: {e}")

            asyncio.create_task(run_ask())

    def handle_clear(b: Any = None) -> None:
        with chat_output:
            clear_output()

    btn_send.on_click(handle_query)
    text_input.on_submit(handle_query)
    btn_clear.on_click(handle_clear)

    chat_ui = widgets.VBox([
        widgets.HBox([text_input, btn_send, btn_clear]),
        chat_output,
    ])
    display(chat_ui)


async def ask(question: str, chatbot: Any | None = None) -> Any:
    """demo.ipynb dosyasındaki gibi programatik olarak asistana soru sorup cevabı ve kaynakları yazdırır.

    Girdi Parametreleri (Inputs):
        question (str): Asistana sorulacak soru metni.
        chatbot (AssistantService | None): Kullanılacak asistan nesnesi.
            Belirtilmezse ortamdaki 'chatbot' değişkenini alır.

    Döndürdüğü Değer (Output):
        AssistantResponse: Asistanın yanıt nesnesi (response.answer, response.citations, response.intent).

    Nasıl Kullanılır:
        >>> await ask("RAG mimarisi nedir?", chatbot=chatbot)
    """
    active_bot = chatbot or globals().get("chatbot")
    if active_bot is None:
        raise ValueError("'chatbot' nesnesi bulunamadı! Lütfen AssistantService nesnenizi sağlayın.")
    response = await active_bot.answer(question)
    if response is None:
        raise ValueError("Asistan yanıt veremedi!")
    display(Markdown(f"**Yanıt:**\n{response.answer}"))
    if getattr(response, "citations", None):
        print(
            active_bot.show_citations(
                citations=response.citations,
                score_field="rerank_score",
            )
        )
    return response


__all__ = [
    "create_upload_widget",
    "save_uploaded_file",
    "create_chat_widget",
    "ask",
]
