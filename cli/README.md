# sys42 CLI

Terminaldan startup g'oyasini tayyor texnik topshiriqqa aylantiradi.

## O'rnatish

```bash
cd backend/cli
pip install -e .
```

`-e` (editable) — kodni o'zgartirsangiz qayta o'rnatish shart emas.
Oddiy o'rnatish uchun `-e` siz `pip install .`.

## Foydalanish

```bash
sys42 login          # hisobga kirish (saytdagi hisobning o'zi)
sys42                # g'oya -> savollar -> texnik topshiriq -> sys42-prompt.md
sys42 show           # natijani terminalda o'qish
sys42 models         # mavjud AI modellar
sys42 whoami         # hisob va tarif
sys42 logout
```

Interaktiv savollarsiz, bitta buyruqda:

```bash
sys42 new --idea "Sartaroshxonalar uchun onlayn navbat tizimi..." -y --out TZ.md
```

## Sozlash

| O'zgaruvchi | Vazifasi |
|---|---|
| `SYS42_API_URL` | Backend manzili (standart: production) |
| `SYS42_HOME` | Sozlama papkasi (standart: `~/.sys42`) |

Lokal backendga qarash:

```bash
SYS42_API_URL=http://localhost:8010 sys42 whoami
```

## Nega API kalit so'ralmaydi

CLI ishning o'zini bajarmaydi — sys42 backendini chaqiradi. ML yadro,
playbook'lar, stack tavsiyalari va model kalitlari serverda turadi. Foydalanuvchi
tomonda faqat hisob tokeni saqlanadi (`~/.sys42/config.json`, huquq 0600).
Limit ham saytdagi tarifning o'zi — CLI va sayt bitta hisobni bo'lishadi.
