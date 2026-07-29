"""`sys42` buyrug'i — terminaldan g'oyadan texnik topshiriqqacha.

Argumentsiz `sys42` yozilsa to'liq oqim ishga tushadi:
g'oya -> aniqlashtiruvchi savollar -> texnik topshiriq -> fayl.

CLI ishning o'zini qilmaydi — sys42 backendini chaqiradi. Shu sababdan
foydalanuvchidan hech qanday API kalit so'ralmaydi: ML yadro, playbook'lar,
stack tavsiyalari va model kalitlari serverda turadi, hisob esa bitta —
saytdagi hisobning o'zi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import config
from .api import Api, ApiError

console = Console()
err = Console(stderr=True)

DEFAULT_MODEL = "anthropic/claude-opus-5"
OUTPUT_FILE = "sys42-prompt.md"


def _api() -> Api:
    return Api(config.load())


def _require_login() -> Api:
    session = config.load()
    if not session.logged_in:
        err.print("[red]Tizimga kirilmagan.[/red] `sys42 login` buyrug'ini bajaring.")
        raise SystemExit(1)
    return Api(session)


def _fail(exc: ApiError) -> None:
    err.print(f"[red]Xato:[/red] {exc}")
    raise SystemExit(1)


# --------------------------------------------------------------------------- #


@click.group(invoke_without_command=True)
@click.version_option(package_name="sys42", prog_name="sys42")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """sys42 — startup g'oyasidan tayyor texnik topshiriqqacha."""
    # Argumentsiz chaqirilsa asosiy oqimga o'tamiz — CLI ning butun mazmuni
    # shu, foydalanuvchi buyruq nomini eslab yurishi shart emas.
    if ctx.invoked_subcommand is None:
        ctx.invoke(new)


@cli.command()
@click.option("--email", prompt="Email", help="sys42 hisobingiz emaili")
@click.option("--password", prompt="Parol", hide_input=True, help="Hisob paroli")
def login(email: str, password: str) -> None:
    """Hisobga kirish. Token `~/.sys42/config.json` da saqlanadi."""
    session = config.load()
    try:
        data = Api(session).login(email, password)
    except ApiError as exc:
        _fail(exc)

    session.token = data.get("access_token", "")
    session.email = (data.get("user") or {}).get("email", email)
    config.save(session)

    user = data.get("user") or {}
    console.print(
        f"[green]Kirdingiz:[/green] {session.email}  "
        f"[dim](tarif: {user.get('plan', '?')})[/dim]"
    )


@cli.command()
def logout() -> None:
    """Saqlangan tokenni o'chiradi."""
    config.clear()
    console.print("Chiqdingiz.")


@cli.command()
def whoami() -> None:
    """Joriy hisob va limit holati."""
    api = _require_login()
    try:
        me = api.me()
    except ApiError as exc:
        _fail(exc)

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_row("Server", api.session.api_url)
    table.add_row("Email", str(me.get("email", "—")))
    table.add_row("Ism", str(me.get("name") or "—"))
    table.add_row("Tarif", str(me.get("plan", "—")))
    table.add_row("Rol", str(me.get("role", "—")))
    console.print(table)


@cli.command()
@click.option("--all", "show_all", is_flag=True, help="Faqat mashhurlarini emas, hammasini")
def models(show_all: bool) -> None:
    """Mavjud AI modellar ro'yxati."""
    try:
        rows = _api().models(only_popular=not show_all)
    except ApiError as exc:
        _fail(exc)

    table = Table(title="AI modellar")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Nomi")
    table.add_column("Kontekst", justify="right")
    table.add_column("$/1M kirish", justify="right")

    for m in rows:
        price = m.get("prompt_usd_per_1m")
        table.add_row(
            str(m.get("id", "")),
            str(m.get("name", "")),
            f"{m.get('context_length', 0):,}",
            "—" if price is None else f"{price:g}",
        )
    console.print(table)


@cli.command()
@click.option("--idea", help="G'oya matni. Berilmasa so'raladi.")
@click.option("--model", "target_model", default=DEFAULT_MODEL, show_default=True,
              help="Texnik topshiriq qaysi model uchun moslashtirilsin")
@click.option("--users", "monthly_users", default=1000, show_default=True,
              help="Kutilayotgan oylik faol foydalanuvchi")
@click.option("--lang", default="uz", type=click.Choice(["uz", "ru", "en"]), show_default=True)
@click.option("--region", default="uz", type=click.Choice(["uz", "eu", "us", "global"]), show_default=True)
@click.option("--out", "out_path", default=OUTPUT_FILE, show_default=True,
              help="Natija qaysi faylga yozilsin")
@click.option("--yes", "-y", is_flag=True, help="Savollarni o'tkazib yuborish")
def new(idea: str | None, target_model: str, monthly_users: int, lang: str,
        region: str, out_path: str, yes: bool) -> None:
    """G'oyadan texnik topshiriq yaratadi va faylga yozadi."""
    api = _require_login()

    if not idea:
        console.print(Panel(
            "Loyiha g'oyangizni yozing — nima qiladi, kim uchun, qanday pul topadi.\n"
            "[dim]Kamida 20 belgi. Tugatgach Enter.[/dim]",
            title="sys42", border_style="cyan",
        ))
        idea = click.prompt("G'oya", type=str).strip()

    if len(idea) < 20:
        err.print("[red]G'oya juda qisqa[/red] — kamida 20 belgi kerak.")
        raise SystemExit(1)

    # 1-qadam: aniqlashtiruvchi savollar. Bu arzon zanjirga boradi.
    answers: list[dict] = []
    if not yes:
        with console.status("Savollar tayyorlanmoqda..."):
            try:
                clarify = api.clarify(idea, lang=lang)
            except ApiError as exc:
                _fail(exc)

        console.print(
            f"\n[bold]{clarify.get('project_title', '')}[/bold] "
            f"[dim]({clarify.get('project_type', '')})[/dim]\n"
        )
        questions = clarify.get("questions") or []
        for i, q in enumerate(questions, 1):
            text = q.get("question", "")
            hint = q.get("hint", "")
            console.print(f"[cyan]{i}/{len(questions)}[/cyan] {text}")
            if hint:
                console.print(f"      [dim]{hint}[/dim]")
            # Bo'sh javob ruxsat etilgan — server javobsizlarini tashlab yuboradi.
            answer = click.prompt("      Javob", default="", show_default=False).strip()
            answers.append({"question": text, "answer": answer})
            console.print()

    # 2-qadam: texnik topshiriq. Bu og'ir zanjir — bir necha daqiqa ketishi mumkin.
    with console.status("Texnik topshiriq yozilmoqda (bir necha daqiqa)..."):
        try:
            result = api.generate(
                description=idea,
                target_model=target_model,
                answers=answers,
                monthly_users=monthly_users,
                region=region,
                lang=lang,
            )
        except ApiError as exc:
            _fail(exc)

    prompt = result.get("prompt") or ""
    if not prompt:
        err.print("[red]Server bo'sh natija qaytardi.[/red]")
        raise SystemExit(1)

    target = Path(out_path)
    if target.exists() and not click.confirm(f"{target} mavjud. Ustiga yozilsinmi?", default=False):
        console.print("Bekor qilindi.")
        return
    target.write_text(prompt, encoding="utf-8")

    analysis = result.get("analysis") or {}
    console.print()
    console.print(Panel(
        f"[green]Tayyor:[/green] {target.resolve()}\n"
        f"Loyiha turi: {analysis.get('project_title') or analysis.get('project_type', '—')}\n"
        f"Model: {result.get('target_model', target_model)}\n"
        f"{'LLM qayta yozdi' if result.get('used_llm') else 'Deterministik skelet'}",
        border_style="green",
    ))
    if result.get("warning"):
        console.print(f"[yellow]Eslatma:[/yellow] {result['warning']}")
    console.print(
        f"\n[dim]Endi shu faylni Claude Code / Cursor ga bering:[/dim]\n"
        f"  claude \"$(cat {target})\"\n"
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False), default=OUTPUT_FILE)
def show(path: str) -> None:
    """Yaratilgan texnik topshiriqni terminalda ko'rsatadi."""
    console.print(Markdown(Path(path).read_text(encoding="utf-8")))


def main() -> None:
    try:
        cli()
    except KeyboardInterrupt:
        err.print("\nBekor qilindi.")
        sys.exit(130)


if __name__ == "__main__":
    main()
