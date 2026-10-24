

import re


def extract_information(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    store = "Not detected"
    date = "Not detected"
    invoice = "Not detected"
    items = []

    # Extract invoice date
    date_match = re.search(
        r"(?:Invoice\s+Date|Date)\s*[:#-]?\s*"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        text,
        re.IGNORECASE
    )

    if date_match:
        date = date_match.group(1)

    # Extract invoice number
    invoice_match = re.search(
        r"(?:Invoice\s*(?:Number|No\.?|#))\s*[:#-]?\s*"
        r"([A-Z0-9-]+)",
        text,
        re.IGNORECASE
    )

    if invoice_match:
        invoice = invoice_match.group(1)

    # Detect a possible store name, avoiding template placeholders
    ignored_store_words = (
        "invoice", "date", "due date", "seller", "bill to",
        "payment details", "description", "subtotal", "total",
        "tax", "discount", "thank you", "your company",
        "client's company", "your phone", "address",
        "your products", "notes"
    )

    for line in lines:
        candidate = line.strip("[]() ")

        if (
            candidate
            and not any(
                word in candidate.lower()
                for word in ignored_store_words
            )
            and not re.search(r"\d", candidate)
            and len(candidate) < 60
        ):
            store = candidate
            break

    # Extract purchased item rows from the invoice table
    for line in lines:
        lower_line = line.lower()

        if any(word in lower_line for word in (
            "description", "subtotal", "total to pay",
            "tax (", "invoice number", "invoice date",
            "due date", "payment details", "thank you"
        )):
            continue

        # Match rows with a description, quantity, and three money values.
        # Example: Your Products 2 Items 1,000.00 2,000.00 200.00
        row_match = re.match(
            r"^(.*?)\s+(\d+)\s*(?:items?)?\s+"
            r"(-?\$?[\d,]+\.\d{2})\s+"
            r"(-?\$?[\d,]+\.\d{2})\s+"
            r"(-?\$?[\d,]+\.\d{2})\s*$",
            line,
            re.IGNORECASE
        )

        if row_match:
            name = row_match.group(1).strip(" []()")
            quantity = int(row_match.group(2))
            unit_price = float(
                row_match.group(3).replace("$", "").replace(",", "")
            )
            subtotal = float(
                row_match.group(4).replace("$", "").replace(",", "")
            )
            tax = float(
                row_match.group(5).replace("$", "").replace(",", "")
            )

            if name and not any(
                word in name.lower()
                for word in ("subtotal", "total", "tax")
            ):
                items.append({
                    "item": name,
                    "quantity": quantity,
                    "price": subtotal,
                    "unit_price": unit_price,
                    "tax": tax
                })
            continue

        # Fallback for simple receipts with one price at line end
        simple_match = re.match(
            r"^(.+?)\s+(-?\$?[\d,]+\.\d{2})$",
            line
        )

        if simple_match:
            name = simple_match.group(1).strip(" []()")
            lower_name = name.lower()

            if any(word in lower_name for word in (
                "subtotal", "total", "tax", "discount",
                "invoice", "date", "phone", "address"
            )):
                continue

            price = float(
                simple_match.group(2).replace("$", "").replace(",", "")
            )

            if name:
                items.append({
                    "item": name,
                    "quantity": 1,
                    "price": price
                })

    return {
        "store": store,
        "date": date,
        "invoice": invoice,
        "items": items
    }
