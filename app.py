import torchaudio
import torch.nn.functional as F
import torch
import numpy as np
from pathlib import Path
import atexit
import tempfile
import shutil
import os
import warnings
import streamlit as st

st.set_page_config(
    page_title="Верификация дикторов",
    page_icon="🎤",
    layout="wide"
)

warnings.filterwarnings('ignore')


# Фикс WinError 1314
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HUGGINGFACE_HUB_DISABLE_SYMLINKS"] = "1"

_original_os_symlink = os.symlink


def _safe_os_symlink(src, dst, *args, **kwargs):
    try:
        _original_os_symlink(src, dst, *args, **kwargs)
    except (OSError, NotImplementedError):
        src_p = Path(src)
        dst_p = Path(dst)
        if dst_p.exists() or dst_p.is_symlink():
            return
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        if src_p.is_dir():
            shutil.copytree(str(src_p), str(dst_p))
        else:
            shutil.copy2(str(src_p), str(dst_p))


os.symlink = _safe_os_symlink

try:
    import huggingface_hub.file_download as _hf_fd

    def _hf_no_symlink(src, dst, *args, **kwargs):
        src_p = Path(src)
        dst_p = Path(dst)
        if dst_p.exists():
            return
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        if src_p.is_dir():
            shutil.copytree(str(src_p), str(dst_p))
        else:
            shutil.copy2(str(src_p), str(dst_p))

    for _attr in dir(_hf_fd):
        if 'symlink' in _attr.lower():
            try:
                setattr(_hf_fd, _attr, _hf_no_symlink)
            except Exception:
                pass
except Exception:
    pass


# Загружаем аудио через soundfile/scipy — без torchaudio и torchcodec

def load_audio(path: str) -> torch.Tensor:
    """
    Читает аудиофайл через soundfile (WAV/FLAC) или pydub (MP3/M4A),
    конвертирует в моно 16кГц тензор.
    """
    import numpy as np
    path = str(path)
    ext = Path(path).suffix.lower()

    try:
        # Попытка 1: soundfile (работает для WAV, FLAC, OGG)
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        # data shape: (samples, channels) → транспонируем в (channels, samples)
        signal = torch.from_numpy(data.T)
    except Exception:
        try:
            # Попытка 2: pydub (работает для MP3, M4A, и всего остального)
            from pydub import AudioSegment
            audio = AudioSegment.from_file(path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples /= 32768.0  # нормализация int16 → float32
            signal = torch.from_numpy(samples).unsqueeze(0)
            sr = 16000
        except Exception as e2:
            raise RuntimeError(
                f"Не удалось прочитать аудиофайл: {e2}\n"
                "Установите: pip install soundfile pydub"
            )

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # Ресемплинг до 16000 Гц если нужно
    if sr != 16000:
        try:
            import torchaudio
            signal = torchaudio.functional.resample(signal, sr, 16000)
        except Exception:
            # Fallback через scipy
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, 16000)
            up, down = 16000 // g, sr // g
            arr = resample_poly(signal.numpy()[0], up, down).astype(np.float32)
            signal = torch.from_numpy(arr).unsqueeze(0)

    return signal


if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None

SpeakerRecognition = None
_speechbrain_error = None
try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:
    try:
        from speechbrain.pretrained import SpeakerRecognition
    except ImportError as e:
        _speechbrain_error = str(e)

# ===== Заголовок =====
st.title("Система верификации дикторов")
st.markdown(
    "**Модель:** ECAPA-TDNN (SpeechBrain) — сравнение двух голосовых образцов "
    "для определения принадлежности одному человеку."
)

if SpeakerRecognition is None:
    st.error(
        f"SpeechBrain не найден.\n\n"
        f"Ошибка: `{_speechbrain_error}`\n\n"
        "Установите: `pip install speechbrain==0.5.16`"
    )
    st.stop()

# ===== Загрузка модели =====
MODEL_DIR = Path(__file__).parent / "ecapa_model_cache"
MODEL_DIR.mkdir(exist_ok=True)


@st.cache_resource(show_spinner=False)
def load_model():
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODEL_DIR),
        run_opts={"device": "cpu"}
    )


with st.spinner("Загрузка модели ECAPA-TDNN..."):
    try:
        verification = load_model()
        st.success("Модель ECAPA-TDNN загружена и готова к работе.")
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        st.info(
            "**Что сделать:**\n"
            "1. Удалите папку `ecapa_model_cache` рядом со скриптом\n"
            "2. Удалите `C:\\Users\\User\\.cache\\huggingface\\hub\\models--speechbrain--spkrec-ecapa-voxceleb`\n"
            "3. Запустите через `run_app.bat`"
        )
        st.stop()


# ===== Верификация через encode_batch =====
def verify_speakers(path1: str, path2: str, threshold: float = 0.25):
    wav1 = load_audio(path1)
    wav2 = load_audio(path2)

    with torch.no_grad():
        emb1 = verification.encode_batch(wav1).squeeze()
        emb2 = verification.encode_batch(wav2).squeeze()

    score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    return score, score >= threshold


# ===== Session state =====
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = {
        'file1': {'type': None, 'data': None, 'sample_rate': None, 'path': None},
        'file2': {'type': None, 'data': None, 'sample_rate': None, 'path': None},
    }
if 'mic_temp_files' not in st.session_state:
    st.session_state.mic_temp_files = []


def save_uploaded_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpf:
        tmpf.write(uploaded_file.getbuffer())
        return tmpf.name


def get_audio_path(idx: int):
    data = st.session_state.audio_data[f'file{idx}']
    if data['type'] == 'file' and data['path'] and os.path.exists(data['path']):
        return data['path']
    if data['type'] == 'mic' and data['data']:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
            tmpf.write(data['data'])
            path = tmpf.name
        st.session_state.mic_temp_files.append(path)
        return path
    return None


def cleanup_temp_files(paths: list):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass


def render_speaker_column(col_label, file_key, recorder_key, data_key):
    st.subheader(col_label)
    tab_file, tab_mic = st.tabs(["📁 Загрузить файл", "🎙️ Записать голос"])

    with tab_file:
        uploaded = st.file_uploader(
            "Выберите аудиофайл",
            type=['wav', 'mp3', 'flac', 'm4a'],
            key=file_key
        )
        if uploaded is not None:
            old_path = st.session_state.audio_data[data_key]['path']
            if old_path and os.path.exists(old_path):
                try:
                    os.unlink(old_path)
                except Exception:
                    pass
            new_path = save_uploaded_to_temp(uploaded)
            st.session_state.audio_data[data_key]['path'] = new_path
            st.session_state.audio_data[data_key]['type'] = 'file'
            st.success(f"Файл «{uploaded.name}» загружен")
            st.audio(uploaded)

    with tab_mic:
        try:
            from streamlit_mic_recorder import mic_recorder
            audio_rec = mic_recorder(
                start_prompt="🔴 Начать запись",
                stop_prompt="⏹️ Остановить",
                format="wav",
                key=recorder_key
            )
            if audio_rec and 'bytes' in audio_rec:
                st.session_state.audio_data[data_key]['data'] = audio_rec['bytes']
                st.session_state.audio_data[data_key]['type'] = 'mic'
                st.session_state.audio_data[data_key]['sample_rate'] = audio_rec.get(
                    'sample_rate', 16000)
                st.success("Голос записан!")
                st.audio(audio_rec['bytes'])
                st.caption(
                    f"Sample Rate: {st.session_state.audio_data[data_key]['sample_rate']} Hz")
        except ImportError:
            st.warning(
                "📌 Для записи с микрофона:\n```\npip install streamlit-mic-recorder\n```")


# ===== Интерфейс =====
col1, col2 = st.columns(2, gap="large")
with col1:
    render_speaker_column("👤 Первый диктор", "file1", "recorder1", "file1")
with col2:
    render_speaker_column("👤 Второй диктор", "file2", "recorder2", "file2")

st.divider()
col_btn, col_res = st.columns([1, 2])
with col_btn:
    verify_btn = st.button("Сравнить дикторов",
                           type="primary", use_container_width=True)

if verify_btn:
    path1 = get_audio_path(1)
    path2 = get_audio_path(2)

    if path1 is None or path2 is None:
        st.warning("Загрузите или запишите аудио для **обоих** дикторов.")
    else:
        with st.spinner("Анализ голосовых отпечатков..."):
            try:
                score_val, is_same = verify_speakers(path1, path2)
                progress_val = float(np.clip((score_val + 1) / 2, 0.0, 1.0))

                with col_res:
                    st.subheader("Результат сравнения:")
                    if is_same:
                        st.success("### ✅ Голоса принадлежат ОДНОМУ человеку")
                        st.balloons()
                    else:
                        st.error("### ❌ Голоса принадлежат РАЗНЫМ людям")

                    st.markdown(f"**Косинусное сходство:** `{score_val:.4f}`")
                    st.progress(progress_val, text="Степень сходства голосов")

                    with st.expander("ℹ️ Подробнее"):
                        st.markdown("""
                        - **Порог:** 0.25 (косинусное сходство)
                        - **Score ≥ 0.25** → один диктор
                        - **Score < 0.25** → разные дикторы
                        - **Модель:** ECAPA-TDNN на VoxCeleb
                        - **Точность:** ~99.3% на VoxCeleb1
                        """)
            except Exception as e:
                st.error(f"Ошибка при обработке аудио: {e}")
                st.info("Убедитесь, что файлы содержат чёткую речь 3–5 секунд.")
            finally:
                cleanup_temp_files(st.session_state.mic_temp_files)
                st.session_state.mic_temp_files = []


def _cleanup_on_exit():
    for i in [1, 2]:
        data = st.session_state.get('audio_data', {}).get(f'file{i}', {})
        if data.get('type') == 'file' and data.get('path'):
            try:
                if os.path.exists(data['path']):
                    os.unlink(data['path'])
            except Exception:
                pass
    for p in st.session_state.get('mic_temp_files', []):
        try:
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass


atexit.register(_cleanup_on_exit)

with st.sidebar:
    st.markdown("""
## 📖 Инструкция

1. **Загрузите файлы** или **запишите голос** для обоих дикторов
2. Нажмите **«Сравнить дикторов»**
3. Получите результат

### 🎯 Рекомендации
- **Длительность:** 3–5 секунд речи
- **Качество:** Чистая запись, без шума
- **Формат:** WAV, MP3, FLAC, M4A

### 🔬 О модели
- **ECAPA-TDNN** — современная архитектура
- **Точность:** ~99.3% на VoxCeleb1
- **Вход:** 16 кГц, моно (конвертация автоматически)
""")
    st.divider()
    st.caption("Курсовая работа: Верификация дикторов с помощью ECAPA-TDNN")
