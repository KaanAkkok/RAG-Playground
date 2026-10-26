import re

_INVISIBLE_CHARS = str.maketrans(
    {
        "\x00": "",
        "\u00ad": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2060": "",
        "\ufeff": "",
        "\u00a0": " ",
        "\u202f": " ",
    }
)
_TEXT_TC_RE = re.compile(r"\b[1-9]\d{10}\b")
_TEXT_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_TEXT_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?90[- ]?)?(?:0[- ]?)?5\d{2}[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}(?!\d)"
)
_TEXT_PLATE_RE = re.compile(
    r"(?<![\w])(?:0[1-9]|[1-7]\d|8[01])\s?[A-Z]{1,3}\s?\d{2,4}(?![\w])", re.IGNORECASE
)
_TEXT_IBAN_RE = re.compile(r"\bTR\s?\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE)
_TEXT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_TEXT_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_TEXT_BIRTH_DATE_RE = re.compile(
    r"(?i)\b(?:do[gğ]um\s*tarihi|do[gğ]um\s*g[uü]n[uü])\s*[:=]?\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"
)
_TEXT_NAME_RE = re.compile(
    r"(?i)\b(?:ad[iı]m|ismim|ad[iı]\s*soyad[iı]m|isim\s*soyisim|soyad[iı]m)\s*[:=]?\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü'-]+(?:\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü'-]+){0,3}"
)
_TEXT_ADDRESS_RE = re.compile(
    r"(?i)\b(?:adresim|ikamet\s*adresim|ev\s*adresim)\s*[:=]?\s*[^\n.!?]{3,160}"
)
_PROMPT_INJECTION_RE = re.compile(
    r"""
    # 1. Talimatları geçersiz kılma
    \b(?:
        ignore|disregard|forget|override|bypass|neglect|
        ignorar|ignora|olvida|omitir|desconsiderar|
        ignorer|oublier|contourner|négliger|
        ignorieren|vergiss|umgehen|missachten|
        ignorare|dimentica|eludere|
        ignorar|esquecer|contornar|
        игнорировать|забудь|обойти|
        تجاهل|انسَ|تجاوز|
        忽略|忘记|无视|
        unut|yoksay|atla|geçersizleştir
    )\b.{0,80}\b(?:
        previous|prior|above|earlier|system|developer|
        instructions?|rules?|prompts?|directives?|
        anteriores?|instrucciones?|reglas?|indicaciones?|
        précédentes?|instructions?|règles?|
        vorherigen|anweisungen|regeln|
        precedenti|istruzioni|regole|
        anteriores|instruções|regras|
        предыдущие|инструкции|правила|
        السابقة|التعليمات|القواعد|
        之前的|指令|规则|
        önceki|talimatlar?|kurallar?|yönergeler?
    )\b

    # 2. Gizli prompt veya talimatları ifşa etme
    | \b(?:
        reveal|expose|show|print|display|repeat| disclose|
        reveal|mostrar|revelar|exponer|imprimir|
        dévoiler|révéler|afficher|répéter|
        enthülle|offenlegen|anzeigen|wiederhole|
        rivela|mostra|esponi|
        revelar|mostrar|expor|
        раскрой|покажи|выведи|
        اكشف|اعرض|أظهر|
        揭示|显示|输出|
        açıkla|göster|ifşa\s+et|yazdır|tekrarla
    )\b.{0,80}\b(?:
        system\s*prompt|developer\s*prompt|hidden\s+prompt|
        system\s+instructions?|internal\s+instructions?|
        secret\s+instructions?|confidential\s+prompt|
        prompt\s+system|instrucciones\s+del\s+sistema|
        instructions?\s+système|systemanweisungen|
        istruzioni\s+di\s+sistema|instruções\s+do\s+sistema|
        системн\w*\s+инструкц\w*|
        تعليمات\s+النظام|系统提示词|系统指令|
        sistem\s+prompt|gizli\s+talimatlar?|
        sistem\s+talimatlar?
    )\b

    # 3. Geliştirici modu ve jailbreak
    | \b(?:
        developer|dev|admin|god|sudo|unrestricted|
        geliştirici|yönetici|sınırsız
    )\s+mode\b
    | \b(?:dan\s+mode|jailbreak|jailbroken|脱狱)\b

    # 4. Rol veya kimlik değiştirme
    | \b(?:
        you\s+are\s+now|act\s+as|pretend\s+to\s+be|
        roleplay\s+as|switch\s+roles?|become|
        ahora\s+eres|actúa\s+como|
        tu\s+es\s+maintenant|agis\s+comme|
        du\s+bist\s+jetzt|tu\s+sei\s+ora|
        você\s+agora\s+é|ты\s+теперь|
        أنت\s+الآن|你现在是|
        sen\s+artık|rol\s+yap|gibi\s+davran|
        kimliğini\s+değiştir
    )\b

    # 5. Güvenlik mekanizmalarını kapatma
    | \b(?:
        disable|deactivate|remove|turn\s+off|circumvent|
        desactivar|deshabilitar|eliminar|
        désactiver|supprimer|désactiver|
        deaktivieren|ausschalten|entfernen|
        disattivare|rimuovere|
        desativar|remover|
        отключить|обойти|удалить|
        تعطيل|إزالة|
        禁用|关闭|移除|
        devre\s+dışı\s+bırak|kapat|kaldır|aş
    )\b.{0,80}\b(?:
        safety|security|guardrails?|filters?|restrictions?|
        policies|protections?|safeguards?|
        seguridad|protecciones|restricciones|
        sécurité|protections|restrictions|
        sicherheit|schutzmaßnahmen|beschränkungen|
        sicurezza|protezioni|restrizioni|
        segurança|proteções|restrições|
        безопасность|защиту|ограничения|
        الأمان|الحماية|القيود|
        安全机制|安全限制|防护措施|
        güvenlik|korumalar?|kısıtlamalar?|filtreler?
    )\b
    
    # 6. Gizli veya hassas bilgileri ifşa etme
    | \b(?:
        gizli|gizlilik|özel|hassas|mahrem|gizlenmiş|
        confidential|secret|hidden|private|sensitive|classified|
        secreto|confidencial|oculto|privado|sensible|
        secret|confidentiel|caché|privé|sensible|
        geheim|vertraulich|verborgen|privat|
        segreto|riservato|nascosto|
        sigiloso|confidencial|oculto|
        секретный|конфиденциальный|скрытый|
        سري|سرية|مخفي|
        机密|秘密|隐藏|隐私
    )\b.{0,60}\b(?:
        ver|veri|bilgi(?:leri)?|açıkla|ifşa\s+et|göster|
        paylaş|söyle|dök|çıkar|ilet|
        reveal|disclose|expose|show|provide|share|tell|print|dump|
        revelar|mostrar|exponer|proporcionar|compartir|
        révéler|divulguer|afficher|fournir|partager|
        enthüllen|offenlegen|anzeigen|bereitstellen|
        rivelare|mostrare|divulgare|fornire|
        revelar|expor|compartilhar|fornecer|
        раскрыть|покажи|предоставить|выведи|
        اكشف|اعرض|شارك|أظهر|
        揭示|显示|提供|泄露
    )\b
    | \b(?:
        ver|göster|açıkla|ifşa\s+et|paylaş|söyle|
        reveal|disclose|expose|provide|show|dump|
        revelar|révéler|offenlegen|rivelare|
        раскрыть|اكشف|揭示
    )\b.{0,60}\b(?:
        gizli\s+bilgiler?|gizli\s+veriler?|
        gizli\s+anahtarlar?|erişim\s+anahtarlar?|
        parolalar?|şifreler?|kimlik\s+bilgileri|
        confidential\s+information|secret\s+data|
        hidden\s+information|private\s+data|
        sensitive\s+information|classified\s+information|
        secrets?|credentials?|passwords?|API\s+keys?|
        access\s+tokens?|private\s+keys?|
        información\s+confidencial|datos\s+secretos|
        informations?\s+confidentielles|données\s+secrètes|
        vertrauliche\s+Informationen|geheime\s+Daten|
        informazioni\s+riservate|dati\s+segreti|
        informações\s+confidenciais|dados\s+secretos|
        конфиденциальные\s+данные|секретная\s+информация|
        المعلومات\s+السرية|البيانات\s+الحساسة|
        机密信息|秘密数据|敏感信息
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_MASKED_PLACEHOLDER = "[MASKED]"
_MASKED_TCKN_PLACEHOLDER = "[MASKED_TCKN]"
_MASKED_EMAIL_PLACEHOLDER = "[MASKED_EMAIL]"
_MASKED_PHONE_PLACEHOLDER = "[MASKED_PHONE]"
_MASKED_PLATE_PLACEHOLDER = "[MASKED_PLATE]"
_MASKED_IBAN_PLACEHOLDER = "[MASKED_IBAN]"
_MASKED_CARD_PLACEHOLDER = "[MASKED_CARD]"
_MASKED_IP_PLACEHOLDER = "[MASKED_IP]"
_MASKED_BIRTH_DATE_PLACEHOLDER = "[MASKED_BIRTH_DATE]"
_MASKED_NAME_PLACEHOLDER = "[MASKED_NAME]"
_MASKED_ADDRESS_PLACEHOLDER = "[MASKED_ADDRESS]"
_MASKED_BIRTH_DATE_PLACEHOLDER = "[MASKED_BIRTH_DATE]"
_PROMP_INJECTION_ANSWER = "Ben bir yapay zeka modeliyim ve güvenlik nedeniyle bu tür talimatları yerine getiremiyorum. Lütfen sorularınızı veya taleplerinizi güvenli ve uygun bir şekilde ifade edin."
MASK_PATTERN_MAP = {
    _TEXT_TC_RE: _MASKED_TCKN_PLACEHOLDER,
    _TEXT_EMAIL_RE: _MASKED_EMAIL_PLACEHOLDER,
    _TEXT_PHONE_RE: _MASKED_PHONE_PLACEHOLDER,
    _TEXT_PLATE_RE: _MASKED_PLATE_PLACEHOLDER,
    _TEXT_IBAN_RE: _MASKED_IBAN_PLACEHOLDER,
    _TEXT_CARD_RE: _MASKED_CARD_PLACEHOLDER,
    _TEXT_IP_RE: _MASKED_IP_PLACEHOLDER,
    _TEXT_BIRTH_DATE_RE: _MASKED_BIRTH_DATE_PLACEHOLDER,
    _TEXT_NAME_RE: _MASKED_NAME_PLACEHOLDER,
    _TEXT_ADDRESS_RE: _MASKED_ADDRESS_PLACEHOLDER,
}


def is_luhn_valid(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(map(int, digits)):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def is_tckn_valid(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 11 or digits[0] == "0":
        return False
    digits = list(map(int, digits))
    odd_sum = sum(digits[i] for i in range(0, 9, 2))
    even_sum = sum(digits[i] for i in range(1, 8, 2))
    check_digit_10 = (odd_sum * 7 - even_sum) % 10
    check_digit_11 = (sum(digits[:10])) % 10
    return check_digit_10 == digits[9] and check_digit_11 == digits[10]


def is_iban_valid(value: str) -> bool:
    iban = re.sub(r"\s+", "", value).upper()
    if not iban.startswith("TR") or len(iban) != 26:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric_iban = ""
    for char in rearranged:
        if char.isdigit():
            numeric_iban += char
        elif char.isalpha():
            numeric_iban += str(ord(char) - 55)
        else:
            return False
    return int(numeric_iban) % 97 == 1
