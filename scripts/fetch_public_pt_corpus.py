#!/usr/bin/env python3
"""Fetch attributed public Portuguese text for the conversa-pt-fluent-v3.1 model.

Sources: Portuguese Wikipedia (CC BY-SA), Project Gutenberg public-domain books
(via the pglaf mirror) and public educational PDFs. Every file is hashed and
described in ``sources_manifest.json``; nothing is downloaded at training time.
"""
import argparse
import csv
import gzip
import hashlib
import html
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = 'TsetlinPortugueseCorpus/3.1 (symbolic corpus; educational use)'
GUTENBERG_MIRROR = 'https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt'
WIKI_API = 'https://pt.wikipedia.org/w/api.php'
PDFS = [
    ('senai_motores_combustao_interna',
     'http://www.escolaelectra.com.br/alumni/biblioteca/Apostila_motores_de_combustao_interna.pdf',
     'SENAI — Mecânica de Automóveis: Motores de Combustão Interna',
     'Publicly accessible educational material; redistribution rights not asserted.'),
]

# Broad core vocabulary: everyday conversation, Brazil, science, technology,
# health, history, arts and engineering. Featured/good articles are added on top.
CURATED_TITLES = """
Brasil|Ciência|História do Brasil|Geografia do Brasil|Língua portuguesa|
Literatura brasileira|Cultura do Brasil|Economia do Brasil|Política do Brasil|
Educação no Brasil|Saúde no Brasil|Meio ambiente do Brasil|Amazônia|Cerrado|
Mata Atlântica|Pantanal|Caatinga|Pampa|Rio Amazonas|Rio São Francisco|
São Paulo|Rio de Janeiro|Brasília|Salvador|Belo Horizonte|Porto Alegre|Recife|
Fortaleza|Curitiba|Manaus|Belém|Goiânia|Florianópolis|Campo Grande|Natal|
João Pessoa|Maceió|Teresina|Cuiabá|Porto Velho|Rio Branco|Boa Vista|Macapá|
Física|Mecânica|Termodinâmica|Energia|Calor|Temperatura|Eletricidade|Magnetismo|
Óptica|Acústica|Gravidade|Teoria da relatividade|Mecânica quântica|
Física de partículas|Astrofísica|Astronomia|Sistema Solar|Sol|Terra|Lua|Marte|
Júpiter|Via Láctea|Universo|Big Bang|Buraco negro|Estrela|Galáxia|
Química|Átomo|Molécula|Elemento químico|Tabela periódica|Ligação química|
Ácido|Base (química)|Reação química|Química orgânica|Bioquímica|Água|Oxigênio|
Carbono|Hidrogênio|Ferro|Petróleo|Combustão|Motor de combustão interna|
Gasolina|Etanol|Diesel|Energia renovável|
Biologia|Célula|DNA|Gene|Evolução|Ecologia|Ecossistema|Biodiversidade|
Botânica|Zoologia|Anatomia|Fisiologia|Genética|Microbiologia|Vírus|Bactéria|
Fungo|Ser humano|Cérebro|Sistema nervoso|Coração|Sangue|Imunidade|Vacina|
Medicina|Doença|Câncer|Diabetes|Nutrição|Exercício físico|Sono|Psicologia|
Emoção|Memória|Inteligência artificial|
Matemática|Número|Aritmética|Álgebra|Geometria|Cálculo|Estatística|
Probabilidade|Função (matemática)|Teorema de Pitágoras|Número primo|Fração|
Equação|Logaritmo|Trigonometria|Matriz (matemática)|Conjunto|Lógica|Algoritmo|
Teoria da computação|Computação|Ciência da computação|Programação de computadores|
Linguagem de programação|Python|C++|JavaScript|Banco de dados|Internet|
World Wide Web|Rede de computadores|Segurança da informação|Software|Hardware|
Aprendizado de máquina|Rede neural artificial|
Filosofia|Filosofia grega|Sócrates|Platão|Aristóteles|Ética|Epistemologia|
Metafísica|Estética|Democracia|Liberalismo|Socialismo|Capitalismo|Comunismo|
Anarquismo|Iluminismo|Existencialismo|Niilismo|Estoicismo|Utilitarismo|
História|Pré-história|Antiguidade|Idade Média|Renascimento|Revolução Industrial|
Revolução Francesa|Primeira Guerra Mundial|Segunda Guerra Mundial|Guerra Fria|
Colonialismo|Escravidão|Império Romano|Grécia Antiga|Egito Antigo|Mesopotâmia|
China|Índia|Japão|Estados Unidos|Europa|África|América Latina|Portugal|Espanha|
França|Alemanha|Itália|Reino Unido|Rússia|
Literatura|Poesia|Romance (gênero)|Conto|Crônica|Teatro|Ensaio|Machado de Assis|
José de Alencar|Clarice Lispector|Guimarães Rosa|Jorge Amado|Fernando Pessoa|
Luís de Camões|Eça de Queiroz|Carlos Drummond de Andrade|Cecília Meireles|
Rachel de Queiroz|Graciliano Ramos|Monteiro Lobato|Paulo Coelho|
Arte|Pintura|Escultura|Arquitetura|Música|Cinema|Fotografia|Dança|Ópera|Design|
Moda|Gastronomia|Culinária do Brasil|Futebol|Capoeira|Samba|Bossa nova|Carnaval|
Linguística|Gramática|Sintaxe|Semântica|Fonética|Morfologia (linguística)|
Tradução|Comunicação|Jornalismo|Publicidade|Retórica|
Engenharia|Engenharia mecânica|Engenharia civil|Engenharia elétrica|
Engenharia química|Robótica|Automóvel|Avião|Barco|Trem|Ponte|Estrada|
Construção civil|Materiais|Aço|Concreto|Plástico|Vidro|Cerâmica|Nanotecnologia|
Impressão 3D|
Antropologia|Sociologia|Ciência política|Direito|Constituição|Direitos humanos|
Cidadania|Globalização|Urbanização|Migração|População|Censo|Desigualdade social|
Pobreza|Racismo|Feminismo|
Religião|Cristianismo|Catolicismo|Protestantismo|Islamismo|Judaísmo|Budismo|
Hinduísmo|Espiritismo|Candomblé|Umbanda|
Saúde pública|Sistema Único de Saúde|Vacinação|Pandemia de COVID-19|COVID-19|
HIV/AIDS|Dengue|Malária|Obesidade|Tabagismo|Álcool|
Esporte|Jogos Olímpicos|Copa do Mundo FIFA|Basquetebol|Voleibol|Natação|
Atletismo|Artes marciais|Xadrez
""".replace('\n', '').strip().split('|')

CATEGORIES = [
    'Categoria:!Artigos destacados',
    'Categoria:!Artigos bons',
]

SKIP_TITLE = re.compile(
    r'(desambiguação|^\d+[\.ºª]|^Lista |^Anexo:|^Categoria:|^Predefinição)', re.I)


def fetch(url, timeout=60, retries=6):
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - network errors vary
            last = error
            # Wikipedia asks clients to slow down (HTTP 429); back off harder.
            delay = min(30, 2 ** (attempt + 1))
            time.sleep(delay)
    raise RuntimeError(f'falha ao baixar {url}: {last}')


def fetch_json(url):
    return json.loads(fetch(url).decode('utf-8'))


def slug(text, limit=90):
    value = re.sub(r'[^\w\-]+', '_', text, flags=re.UNICODE).strip('_')
    return value[:limit] or hashlib.sha256(text.encode()).hexdigest()[:12]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def wiki_category_members(category, limit):
    titles = []
    params = {'action': 'query', 'format': 'json', 'list': 'categorymembers',
              'cmtitle': category, 'cmnamespace': 0, 'cmlimit': str(min(limit, 500))}
    while len(titles) < limit:
        data = fetch_json(WIKI_API + '?' + urllib.parse.urlencode(params))
        titles += [m['title'] for m in data.get('query', {}).get('categorymembers', [])]
        cont = data.get('continue')
        if not cont:
            break
        params.update(cont)
    return titles[:limit]


def drop_balanced(text, opening, closing):
    """Remove ``opening ... closing`` blocks, honouring nesting."""
    output, index = [], 0
    while index < len(text):
        if text.startswith(opening, index):
            depth, index = 1, index + len(opening)
            while index < len(text) and depth:
                if text.startswith(opening, index):
                    depth, index = depth + 1, index + len(opening)
                elif text.startswith(closing, index):
                    depth, index = depth - 1, index + len(closing)
                else:
                    index += 1
        else:
            output.append(text[index])
            index += 1
    return ''.join(output)


def clean_wikitext(text):
    """Convert MediaWiki markup into readable plain text (best effort)."""
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)
    text = re.sub(r'<ref[^>]*/>', ' ', text, flags=re.I)
    text = re.sub(r'<ref[^>]*>.*?</ref>', ' ', text, flags=re.I | re.S)
    text = re.sub(r'<(gallery|math|timeline|score|imagemap|mapframe).*?</\1>', ' ',
                  text, flags=re.I | re.S)
    text = drop_balanced(text, '{{', '}}')
    text = drop_balanced(text, '{|', '|}')
    text = re.sub(r'\[\[(?:Ficheiro|File|Imagem|Image|Categoria|Category):[^\]]*\]\]',
                  ' ', text, flags=re.I)
    text = re.sub(r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]*\]', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r"'{2,5}", '', text)
    text = re.sub(r'^=+\s*(.*?)\s*=+\s*$', r'\1.', text, flags=re.M)
    text = re.sub(r'^\s*[\*#:;]+\s*', '', text, flags=re.M)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()
        if line and not re.fullmatch(r'[\W_]+', line):
            lines.append(line)
    return '\n\n'.join(lines)


def wiki_batch_contents(titles):
    params = {'action': 'query', 'format': 'json', 'prop': 'revisions',
              'rvprop': 'ids|timestamp|content', 'rvslots': 'main',
              'redirects': 1, 'maxlag': 5, 'titles': '|'.join(titles)}
    data = fetch_json(WIKI_API + '?' + urllib.parse.urlencode(params))
    articles = []
    for page in data.get('query', {}).get('pages', {}).values():
        revision = (page.get('revisions') or [{}])[0]
        content = revision.get('slots', {}).get('main', {}).get('*', '')
        if 'missing' in page or not content or content.lstrip().upper().startswith('#REDIRECT'):
            continue
        text = clean_wikitext(content)
        if len(text) < 1200:
            continue
        articles.append({
            'title': page['title'], 'pageid': page.get('pageid'),
            'revision': revision.get('revid'), 'timestamp': revision.get('timestamp'),
            'url': 'https://pt.wikipedia.org/wiki/' + urllib.parse.quote(
                page['title'].replace(' ', '_')),
            'license': 'CC BY-SA 4.0', 'text': text})
    return articles


def run_wikipedia(raw, style, limit):
    titles, seen = [], set()
    for title in CURATED_TITLES:
        if title not in seen:
            seen.add(title)
            titles.append(title)
    for category in CATEGORIES:
        try:
            members = wiki_category_members(category, min(limit, 500))
        except Exception as error:  # noqa: BLE001 - keep curated list on failure
            print(f'[wikipedia] categoria {category}: {error}', flush=True)
            members = []
        for title in members:
            if title not in seen and not SKIP_TITLE.search(title):
                seen.add(title)
                titles.append(title)
    titles = titles[:limit]
    print(f'[wikipedia] {len(titles)} artigos em lotes de 25', flush=True)

    entries, failures = [], 0
    raw_dir, style_dir = raw / 'wikipedia_pt', style / 'wikipedia_pt'
    raw_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 25
    for start in range(0, len(titles), batch_size):
        batch = titles[start:start + batch_size]
        try:
            articles = wiki_batch_contents(batch)
        except Exception as error:  # noqa: BLE001
            failures += len(batch)
            print(f'[wikipedia] lote {start}: {error}', flush=True)
            time.sleep(10)
            continue
        for article in articles:
            name = slug(article['title'])
            text_path = style_dir / f'{name}.txt'
            text_path.write_text(article['text'] + '\n', encoding='utf-8')
            metadata = {key: article[key] for key in
                        ('title', 'pageid', 'revision', 'timestamp', 'url', 'license')}
            (raw_dir / f'{name}.json').write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            entries.append(metadata | {
                'text_file': str(text_path.relative_to(style.parent)),
                'sha256': sha256(text_path.read_bytes()),
                'characters': len(article['text'])})
        print(f'[wikipedia] {min(start + batch_size, len(titles))}/{len(titles)} '
              f'(ok={len(entries)})', flush=True)
        time.sleep(1.5)
    print(f'[wikipedia] ok={len(entries)} falhas={failures}', flush=True)
    return entries, failures


def read_gutenberg_catalog(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', errors='replace', newline='') as handle:
        for row in csv.DictReader(handle):
            language = (row.get('Language') or '').strip().casefold()
            if language == 'pt' or 'portuguese' in language or language.startswith('pt,'):
                yield row


def strip_gutenberg(text):
    start = re.search(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
                      text, re.I | re.S)
    end = re.search(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
                    text, re.I | re.S)
    body = text[start.end() if start else 0:end.start() if end else len(text)]
    return re.sub(r'\s+', ' ', body).strip()


def choose_books(catalog, limit):
    """Pick author-diverse books spread across the whole Portuguese catalogue."""
    rows = sorted(read_gutenberg_catalog(catalog), key=lambda r: int(r['Text#']))
    selected, seen_titles, per_author = [], set(), {}
    for row in rows:
        title_key = re.sub(r'\b(Tomo|Vol\.?|Volume|Parte)\b.*', '', row['Title'],
                           flags=re.I).strip().casefold()
        author = (row.get('Authors') or 'anônimo').split(';')[0].strip().casefold()
        if title_key in seen_titles or per_author.get(author, 0) >= 2:
            continue
        seen_titles.add(title_key)
        per_author[author] = per_author.get(author, 0) + 1
        selected.append(row)
    if len(selected) > limit:
        if limit == 1:
            return selected[:1]
        indexes = [round(i * (len(selected) - 1) / (limit - 1)) for i in range(limit)]
        selected = [selected[i] for i in dict.fromkeys(indexes)]
    return selected


def run_gutenberg(raw, style, catalog, limit):
    raw_dir, style_dir = raw / 'gutenberg', style / 'gutenberg'
    raw_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)
    rows = choose_books(catalog, limit)
    print(f'[gutenberg] {len(rows)} livros escolhidos (limite {limit})', flush=True)

    def download(row):
        identifier = int(row['Text#'])
        url = GUTENBERG_MIRROR.format(id=identifier)
        name = f"{identifier:05d}_{slug(row['Title'])}"
        raw_path = raw_dir / (name + '.txt')
        if raw_path.exists():
            data = raw_path.read_bytes()
        else:
            data = fetch(url, timeout=90)
            raw_path.write_bytes(data)
        text = data.decode('utf-8', errors='replace')
        body = strip_gutenberg(text)
        return row, url, data, body, name

    entries, failures = [], 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download, row): row for row in rows}
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                row, url, data, body, name = future.result()
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"[gutenberg] falhou {row.get('Text#')}: {error}", flush=True)
                continue
            words = len(body.split())
            if words < 4000:
                continue
            text_path = style_dir / (name + '.txt')
            text_path.write_text(body + '\n', encoding='utf-8')
            entries.append({'id': int(row['Text#']), 'title': row['Title'],
                            'authors': row.get('Authors', ''), 'language': row.get('Language', ''),
                            'issued': row.get('Issued', ''), 'url': url,
                            'license': 'Public domain in the United States (Project Gutenberg)',
                            'text_file': str(text_path.relative_to(style.parent)),
                            'sha256': sha256(data), 'characters': len(body), 'words': words})
            if done % 20 == 0:
                print(f'[gutenberg] {done}/{len(rows)}', flush=True)
    print(f'[gutenberg] ok={len(entries)} falhas={failures}', flush=True)
    return entries, failures


def run_pdfs(raw, style):
    raw_dir, style_dir = raw / 'pdfs', style / 'pdfs'
    raw_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)
    entries, failures = [], 0
    for name, url, title, license_note in PDFS:
        try:
            data = fetch(url, timeout=120)
            pdf_path = raw_dir / (name + '.pdf')
            pdf_path.write_bytes(data)
            result = subprocess.run(['pdftotext', '-layout', str(pdf_path), '-'],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or 'pdftotext falhou')
            body = re.sub(r'[ \t]+', ' ', result.stdout)
            body = re.sub(r'\n{3,}', '\n\n', body).strip()
            if len(body.split()) < 500:
                raise RuntimeError('texto extraído muito curto')
            text_path = style_dir / (name + '.txt')
            text_path.write_text(body + '\n', encoding='utf-8')
            entries.append({'title': title, 'url': url, 'license': license_note,
                            'text_file': str(text_path.relative_to(style.parent)),
                            'sha256': sha256(data), 'characters': len(body),
                            'words': len(body.split())})
            print(f'[pdf] ok {title} ({len(body.split())} palavras)', flush=True)
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f'[pdf] falhou {url}: {error}', flush=True)
    return entries, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True,
                        help='dataset root; text goes to <out>/style and raw files to <out>/raw')
    parser.add_argument('--wikipedia-limit', type=int, default=700)
    parser.add_argument('--gutenberg-catalog', type=Path, default=None)
    parser.add_argument('--gutenberg-limit', type=int, default=90)
    parser.add_argument('--skip-wikipedia', action='store_true')
    parser.add_argument('--skip-gutenberg', action='store_true')
    parser.add_argument('--skip-pdfs', action='store_true')
    args = parser.parse_args()

    raw, style = args.out / 'raw', args.out / 'style'
    raw.mkdir(parents=True, exist_ok=True)
    style.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'sources_manifest.json'
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'purpose': 'public Portuguese text for conversa-pt-fluent-v3.1',
                'neural_training': False,
                'sources': {'wikipedia_pt': [], 'gutenberg': [], 'pdfs': []}}
    for group in manifest['sources']:
        manifest['sources'][group] = previous.get('sources', {}).get(group, [])

    if not args.skip_wikipedia:
        entries, failures = run_wikipedia(raw, style, args.wikipedia_limit)
        manifest['sources']['wikipedia_pt'] = entries
        manifest['wikipedia_failures'] = failures
    if not args.skip_gutenberg:
        if not args.gutenberg_catalog:
            parser.error('--gutenberg-catalog é necessário sem --skip-gutenberg')
        entries, failures = run_gutenberg(raw, style, args.gutenberg_catalog,
                                          args.gutenberg_limit)
        manifest['sources']['gutenberg'] = entries
        manifest['gutenberg_failures'] = failures
    if not args.skip_pdfs:
        entries, failures = run_pdfs(raw, style)
        manifest['sources']['pdfs'] = entries
        manifest['pdf_failures'] = failures

    manifest['totals'] = {
        'wikipedia_articles': len(manifest['sources']['wikipedia_pt']),
        'gutenberg_books': len(manifest['sources']['gutenberg']),
        'pdfs': len(manifest['sources']['pdfs']),
        'characters': sum(e['characters'] for group in manifest['sources'].values()
                          for e in group),
        'words': sum(e.get('words', e['characters'] // 6) for group in manifest['sources'].values()
                     for e in group),
    }
    (args.out / 'sources_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest['totals'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
