import tkinter as tk
from tkinter import ttk, filedialog
import pyttsx3
from googletrans import Translator
import glob
from transformers import MarianMTModel, MarianTokenizer
import argostranslate.package, argostranslate.translate
import urllib.request
import os
import json
from vosk import Model, KaldiRecognizer
import sounddevice as sd

class VoskRecognizer:
    def __init__(self, vosk_base_path, vosk_lang_override):
        self.vosk_base_path = vosk_base_path
        self.vosk_lang_override = vosk_lang_override
        self.vosk_models = {}

    def load_model(self, lang_code):
        model_suffix = self.vosk_lang_override.get(lang_code)
        if not model_suffix:
            print(f"No Vosk model override for language code '{lang_code}'")
            return None

        if model_suffix in self.vosk_models:
            return self.vosk_models[model_suffix]

        model_dir = os.path.join(self.vosk_base_path, f"vosk-model-{model_suffix}")
        if not os.path.exists(model_dir):
            print(f"Vosk model directory not found: {model_dir}")
            return None

        try:
            model = Model(model_dir)
            self.vosk_models[model_suffix] = model
            print(f"Loaded Vosk model: {model_dir}")
            return model
        except Exception as e:
            print(f"Error loading Vosk model {model_dir}: {e}")
            return None

    def recognize_offline(self, lang_code, duration=5):
        model = self.load_model(lang_code)
        if not model:
            return None, f"No offline Vosk model for language '{lang_code}'."

        samplerate = 16000
        rec = KaldiRecognizer(model, samplerate)
        rec.SetWords(True)

        recognized_text = []

        def callback(indata, frames, time, status):
            if status:
                print(status)
            data_bytes = bytes(indata)
            if rec.AcceptWaveform(data_bytes):
                result_json = rec.Result()
                result = json.loads(result_json)
                text = result.get("text", "")
                if text:
                    recognized_text.append(text)

        try:
            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16', channels=1, callback=callback):
                sd.sleep(duration * 1000)
        except Exception as e:
            return None, f"Error with microphone or recording: {e}"

        final_json = rec.FinalResult()
        final_result = json.loads(final_json)
        final_text = final_result.get("text", "")

        all_text = " ".join(recognized_text).strip()
        if final_text:
            all_text = (all_text + " " + final_text).strip()

        return all_text, None

class MarianTranslator:
    def __init__(self, base_path):
        self.base_path = base_path
        self.marian_models = {}

    def load_model(self, src_code, tgt_code):
        model_path = os.path.join(self.base_path, f"{src_code}-{tgt_code}")
        if model_path not in self.marian_models:
            if not os.path.exists(model_path):
                print(f"Directory does not exist: {model_path}")
                return None, None
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_path)
                model = MarianMTModel.from_pretrained(model_path)
                self.marian_models[model_path] = (model, tokenizer)
            except Exception as e:
                print(f"Failed to load MarianMT model from {model_path}: {e}")
                return None, None
        return self.marian_models[model_path]

    def translate(self, text, model, tokenizer):
        batch = tokenizer.prepare_seq2seq_batch(src_texts=[text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        translated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return translated

class ArgosTranslator:
    def __init__(self):
        self.load_argos_models()

    def load_argos_models(self):
        model_paths = glob.glob("*.argosmodel")
        for model_path in model_paths:
            try:
                argostranslate.package.install_from_path(model_path)
            except Exception as e:
                print(f"[Argos load error for {model_path}]: {e}")

    def translate(self, text, src_code, tgt_code):
        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next((lang for lang in installed_languages if lang.code == src_code), None)
        to_lang = next((lang for lang in installed_languages if lang.code == tgt_code), None)

        if from_lang and to_lang:
            translation = from_lang.get_translation(to_lang)
            return translation.translate(text)
        else:
            return None

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌐 Multilingual Voice Translator")
        self.root.geometry("700x500")
        self.is_dark = True

        self.lang_map = {
            "Arabic": "ar", "Chinese": "zh", "Dutch": "nl", "English": "en", "Esperanto": "eo",
            "Filipino": "tl", "French": "fr", "German": "de", "Hindi": "hi",
            "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt", "Russian": "ru",
            "Spanish": "es", "Swedish": "sv", "Ukrainian": "uk",
            "Polish": "pl", "Czech": "cs", "Finnish": "fi", "Turkish": "tr"
        }

        self.vosk_lang_override = {
            "en": "small-en-us-0.15",
            "ar": "ar-mgb2-0.4",
            "zh": "small-cn-0.22",
            "de": "small-de-0.15",
            "eo": "small-eo-0.42",
            "es": "small-es-0.42",
            "fr": "small-fr-0.22",
            "hi": "small-hi-0.22",
            "it": "small-it-0.22",
            "ja": "small-ja-0.22",
            "ko": "small-ko-0.22",
            "nl": "small-nl-0.22",
            "pt": "small-pt-0.3",
            "ru": "small-ru-0.22",
            "sv": "small-sv-rhasspy-0.15",
            "uk": "small-uk-v3-small",
            "spk": "spk-0.4",
            "tl": "tl-ph-generic-0.6"
        }

        self.vosk_base_path = "C:/Users/VENNELA/project"
        self.vosk_recognizer = VoskRecognizer(self.vosk_base_path, self.vosk_lang_override)

        self.marian_base_path = r"C:\Users\VENNELA\Desktop\New folder (2)"
        self.marian_translator = MarianTranslator(self.marian_base_path)

        self.argos_translator = ArgosTranslator()

        self.engine = pyttsx3.init()
        self.translator = Translator()  # Google Translate (unused fallback)

        self.setup_theme()
        self.create_widgets()

    def setup_theme(self):
        if self.is_dark:
            self.bg_color = "#2C3E50"
            self.fg_color = "white"
            self.button_bg = "#3498DB"
            self.text_bg = "#34495E"
            self.font_entry = ("Verdana", 12)
            self.entry_bg = "#2c3e50"
            self.entry_fg = "white"
        else:
            self.bg_color = "#ECF0F1"
            self.fg_color = "#2C3E50"
            self.button_bg = "#1ABC9C"
            self.text_bg = "white"
            self.font_entry = ("Verdana", 12)
            self.entry_bg = "white"
            self.entry_fg = "#2c3e50"

        self.root.configure(bg=self.bg_color)

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.setup_theme()
        self.clear_widgets()
        self.create_widgets()

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_widgets(self):
        tk.Label(self.root, text="🌐 Multilingual Voice Translator",
                 font=("Verdana", 20, "bold"), bg=self.bg_color, fg=self.fg_color).pack(pady=(15, 5))

        tk.Button(self.root, text="🌙 Toggle Theme", command=self.toggle_theme,
                  font=("Verdana", 11, "bold"), bg=self.button_bg, fg="white").pack(pady=5)

        lang_frame = tk.Frame(self.root, bg=self.bg_color)
        lang_frame.pack(pady=10)

        tk.Label(lang_frame, text="From:", font=("Verdana", 12), bg=self.bg_color, fg=self.fg_color).grid(row=0, column=0, padx=10)
        self.src_lang_combo = ttk.Combobox(lang_frame, values=list(self.lang_map.keys()), state="readonly", width=15)
        self.src_lang_combo.set("English")
        self.src_lang_combo.grid(row=0, column=1, padx=10)

        tk.Label(lang_frame, text="To:", font=("Verdana", 12), bg=self.bg_color, fg=self.fg_color).grid(row=0, column=2, padx=10)
        self.tgt_lang_combo = ttk.Combobox(lang_frame, values=list(self.lang_map.keys()), state="readonly", width=15)
        self.tgt_lang_combo.set("Hindi")
        self.tgt_lang_combo.grid(row=0, column=3, padx=10)

        self.text_entry = tk.Text(self.root, height=4, font=self.font_entry, bg=self.entry_bg, fg=self.entry_fg)
        self.text_entry.pack(pady=10, padx=20, fill="both", expand=True)

        btn_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="🎤 Speak (Offline Vosk)", font=("Verdana", 11, "bold"), width=18,
                  command=self.speak_input, bg=self.button_bg, fg="white").grid(row=0, column=0, padx=10)

        tk.Button(btn_frame, text="📂 Load File", font=("Verdana", 11, "bold"), width=12,
                  command=self.file_input, bg=self.button_bg, fg="white").grid(row=0, column=1, padx=10)

        tk.Button(btn_frame, text="🌍 Translate", font=("Verdana", 11, "bold"), width=14,
                  command=self.translate_text, bg="#27AE60", fg="white").grid(row=0, column=2, padx=10)

        output_frame = tk.LabelFrame(self.root, text="Translation Output", font=("Verdana", 12),
                                     bg=self.bg_color, fg=self.fg_color)
        output_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.output_label = tk.Label(output_frame, text="", font=("Verdana", 14),
                                     bg=self.bg_color, fg=self.fg_color, wraplength=650, justify="left")
        self.output_label.pack(padx=10, pady=10, fill="both", expand=True)

    def speak_input(self):
        src_lang = self.src_lang_combo.get()
        lang_code = self.lang_map.get(src_lang)
        if not lang_code:
            self.output_label.config(text="Please select a source language supported for speech recognition.")
            return

        self.output_label.config(text="Listening (offline Vosk)... Please speak now.")
        recognized_text, error = self.vosk_recognizer.recognize_offline(lang_code)
        if error:
            self.output_label.config(text=error)
        elif recognized_text:
            self.text_entry.delete("1.0", tk.END)
            self.text_entry.insert(tk.END, recognized_text)
            self.output_label.config(text="Recognized (offline): " + recognized_text)
        else:
            self.output_label.config(text="Could not recognize any speech.")

    def file_input(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                self.text_entry.delete("1.0", tk.END)
                self.text_entry.insert(tk.END, text)

    def translate_text(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        if not text:
            self.output_label.config(text="Please enter or speak some text to translate.")
            return

        src_lang = self.src_lang_combo.get()
        tgt_lang = self.tgt_lang_combo.get()

        if src_lang == tgt_lang:
            self.output_label.config(text="Source and target languages must be different.")
            return

        src_code = self.lang_map.get(src_lang)
        tgt_code = self.lang_map.get(tgt_lang)

        if not src_code or not tgt_code:
            self.output_label.config(text="Please select valid source and target languages.")
            return

        # Try MarianMT offline translation first
        model, tokenizer = self.marian_translator.load_model(src_code, tgt_code)
        if model and tokenizer:
            translated = self.marian_translator.translate(text, model, tokenizer)
            self.output_label.config(text=translated)
            self.speak_text(translated)
            return

        # Else fallback to Argos Translate offline
        translated = self.argos_translator.translate(text, src_code, tgt_code)
        if translated:
            self.output_label.config(text=translated)
            self.speak_text(translated)
        else:
            self.output_label.config(text="Offline translation model not found. Please check your models.")

    def speak_text(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
