"""
characters_config.py
--------------------
Configuração EDITÁVEL da resolução de entidades (03_entity_resolution.py).

Como usar:
  1. Corre 02_ner.py e 03_entity_resolution.py.
  2. Abre Data/unresolved_mentions.csv e Data/candidates_capitalised.csv.
  3. Para cada menção frequente, acrescenta a forma ao nome certo, na coluna
     `aliases` de Data/os_maias_ground_truth_nodes.csv (ou cria uma personagem nova).
     Volta a correr o 03.
  4. Formas ambíguas (o mesmo nome para pessoas diferentes) vão para CONTEXT_RULES.

Atenção: o dicionário vem pré-preenchido. As personagens da secção A foram vistas
no texto (capítulos I-III). As da secção B vêm da memória e têm de ser confirmadas
no texto antes de se confiar nelas.
"""

# Último capítulo em que "Maria Eduarda" é a avó de Carlos (D. Maria Eduarda Runa).
# CONFIRMA com:  python 03_entity_resolution.py --show "Maria Eduarda"
RUNA_LAST_CHAPTER = 2

# Até este capítulo, "Vilaça" sem mais é o pai; depois, o filho. Heurística: confirma.
VILACA_PAI_LAST_CHAPTER = 3

MANUEL_MONFORTE_LAST_CHAPTER = 3

# Palavras que se retiram do início de uma menção antes de a comparar
ARTICLES = {"o", "a", "os", "as", "um", "uma"} # falta do da?
TITLE_TOKENS = {"sr", "sra", "d", "dr", "dra", "dom", "dona", "senhor", "senhora", "mr", "mrs"}

# Nome canónico -> outras formas como aparece no texto.
# (O próprio nome canónico já conta como forma; não é preciso repeti-lo.)
# Deprecated at runtime: aliases are loaded from the `aliases` column in
# Data/os_maias_ground_truth_nodes.csv. Retained as migration/reference data.
ALIASES = {
    # ===================== A. Vistas no texto (capítulos I-III) =====================
    "Caetano da Maia": ["Caetano"],
    "Afonso da Maia": ["Afonso"],
    "D. Maria Eduarda Runa": ["Maria Eduarda Runa"],
    "Pedro da Maia": ["Pedro", "Pedrinho"],
    "Maria Monforte": [],
    "Manuel Monforte": ["papá Monforte"],
    "Carlos da Maia": ["Carlos", "Carlinhos", "Carlos Eduardo", "Carlos de Mata-Sete"],
    "Vilaça (pai)": ["velho Vilaça"],
    "Vilaça Júnior": ["Manuel Vilaça"],          # provavelmente a mesma pessoa; confirma
    "Tomás de Alencar": ["Alencar"],
    "Teixeira": [],
    "Gertrudes": [],
    "Mr. Brown": ["Brown"],
    "Abade Custódio": ["Custódio", "padre Custódio"],
    "Padre Vasques": ["Vasques", "reverendo Vasques"],
    "Coronel Sequeira": ["Sequeira", "General Sequeira"],
    "Tancredo": [],
    "Reverendo Bonifácio": ["Bonifácio", "D. Bonifácio de Calatrava"],
    "Tobias": [],
    "Eusébiozinho": ["Eusebiozinho", "Eusébio"],
    "D. Eugénia Silveira": ["Eugénia"],
    "D. Ana Silveira": ["Ana Silveira"],
    "Teresinha": [],
    "Tia Fanny": ["Fanny"],
    "D. João da Cunha": ["João da Cunha"],
    "D. João Coutinho": ["João Coutinho"],
    "Melo": [],
    "Dr. Guedes": ["Guedes"],
    "André da Ega": [],
    "Luís Runa": [],
    "Conde de Runa": [],
    "Sebastião da Maia": [],
    "D. Maria da Gama": ["Maria da Gama"],
    "Esteves": [],
    "Cazoti": [],

    # ============ B. Da memória: CONFIRMAR no texto (ver candidates_capitalised.csv) ============
    "João da Ega": ["Ega"],
    "Guilherme Craft": ["Craft"],
    "Cruges": [],
    "Dâmaso Salcede": ["Dâmaso", "Salcede"],
    "Maria Eduarda": [],                           # a amante de Carlos (ver CONTEXT_RULES)
    "Rosa": [],
    "Mac Gren": ["Mac-Gren"],
    "Castro Gomes": [],
    "Sr. Guimarães": ["Guimarães"],
    "Conde de Gouvarinho": [],
    "Condessa de Gouvarinho": [],
    "Jacob Cohen": ["Jacob"],
    "Raquel Cohen": ["Raquel"],
    "Steinbroken": [],
    "Sousa Neto": [],
    "Rufino": [],
    "Marquês de Souselas": ["Souselas"],
    "Teles da Gama": [],
    "Palma Cavalão": ["Cavalão"],
    "Melanie": [],
    "Miss Sara": ["Sara"],
    "Domingos": [],
    "Taveira": [],
    "Soveral": [],
    "D. Maria da Cunha": ["Maria da Cunha"],
}

# Formas AMBÍGUAS. A chave é a menção já normalizada (minúsculas, sem acentos,
# sem pontuação, sem artigo nem título). Tipos de regra:
#   chapter   : capítulo <= upto -> "before", senão -> "after" (None = fica por resolver)
#   prev_word : decide pela palavra imediatamente antes da menção no texto
CONTEXT_RULES = {
    # D. Maria Eduarda Runa (avó) vs Maria Eduarda (amante de Carlos)
    "maria eduarda": {
        "type": "chapter", "upto": RUNA_LAST_CHAPTER,
        "before": "Maria Eduarda Runa", "after": "Maria Eduarda",
    },
    # Vilaça pai vs filho
    "vilaca": {
        "type": "chapter", "upto": VILACA_PAI_LAST_CHAPTER,
        "before": "Vilaça", "after": "Vilaça Júnior",
    },

    "manuel": {
            "type": "chapter", "upto": MANUEL_MONFORTE_LAST_CHAPTER,
            "before": "Manuel Monforte", "after": "Vilaça Júnior",
        },

    # "a Monforte" = Maria; "o Monforte", "papá Monforte", "o velho Monforte" = Manuel
    "monforte": {
        "type": "prev_word",
        "map": {"a": "Maria Monforte", "o": "Manuel Monforte", "papa": "Manuel Monforte",
                "velho": "Manuel Monforte"},
        "default": "Manuel Monforte",
    },
    # "Maria" sozinho: Maria Monforte nos primeiros capítulos; depois fica por resolver
    "maria": {"type": "chapter", "upto": 10, "before": "Maria Monforte", "after": "Maria Eduarda"},
    # Gouvarinho: conde ou condessa (da memória; confirma)
    "gouvarinho": {
        "type": "prev_word",
        "map": {"o": "Conde de Gouvarinho", "conde": "Conde de Gouvarinho", "do": "Conde de Gouvarinho", "conde de Gouvarinho": "Conde de Gouvarinho",
                "a": "Condessa de Gouvarinho", "condessa": "Condessa de Gouvarinho", "da": "Condessa de Gouvarinho", "boa": "Condessa de Gouvarinho", "condessa de Gouvarinho": "Condessa de Gouvarinho", "na Gouvarinho": "Condessa de Gouvarinho"},
        "default": None,
    },
    # Cohen: Jacob ou Raquel (da memória; confirma)
    "cohen": {
        "type": "prev_word",
        "map": {"a": "Raquel Cohen", "o": "Jacob Cohen", "da": "Raquel Cohen", "do": "Jacob Cohen"},
        "default": "Jacob Cohen",
    },

    "castros gomes": {
    "type": "multiple",
    "characters": ["Joaquim Álvares de Castro Gomes", "Maria Eduarda"],
},

    "gouvarinhos": {
    "type": "multiple",
    "characters": ["Conde de Gouvarinho", "Condessa de Gouvarinho"],

},
    "cohens": {
    "type": "multiple",
    "characters": ["Jacob Cohen", "Raquel Cohen"],
   
    },

    "monfortes": {
    "type": "multiple",
    "characters": ["Maria Monforte", "Manuel Monforte"],
   
    },

}
