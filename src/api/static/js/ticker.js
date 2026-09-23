/**
 * ticker.js
 *
 * Builds the scrolling ticker strip from page-provided JSON.
 * Kept standalone for browser caching across page navigations.
 */

(function () {
    "use strict";

    function formatPrice(value) {
        return "$" + Number(value).toFixed(2);
    }

    function formatChange(value) {
        const sign = value > 0 ? "+" : "";
        return sign + Number(value).toFixed(2) + "%";
    }

    function changeClass(value) {
        if (value > 0) return "is-gain";
        if (value < 0) return "is-loss";
        return "is-flat";
    }

    function buildTickerItem(entry) {
        const item = document.createElement("span");
        item.className = "ticker-item";

        const symbol = document.createElement("span");
        symbol.className = "ticker-symbol";
        symbol.textContent = entry.symbol;

        const price = document.createElement("span");
        price.className = "ticker-price";
        price.textContent = entry.current_price !== null ? formatPrice(entry.current_price) : "—";

        const change = document.createElement("span");
        change.className = "ticker-change " + changeClass(entry.daily_change_pct || 0);
        change.textContent = entry.daily_change_pct !== null ? formatChange(entry.daily_change_pct) : "—";

        item.appendChild(symbol);
        item.appendChild(price);
        item.appendChild(change);
        return item;
    }

    function init() {
        const dataEl = document.getElementById("ticker-data");
        const track = document.getElementById("ticker-track");
        if (!dataEl || !track) {
            return;
        }

        let entries = [];
        try {
            entries = JSON.parse(dataEl.textContent);
        } catch (err) {
            return; // Leave the default "connect a watchlist" message in place.
        }

        const tickerEntries = entries.filter(function (e) {
            return e.current_price !== null && e.current_price !== undefined;
        });

        if (tickerEntries.length === 0) {
            return; // Leave the default empty-state message in place.
        }

        track.innerHTML = "";
        // Duplicate the list so the marquee loop has no visible seam.
        const doubled = tickerEntries.concat(tickerEntries);
        doubled.forEach(function (entry) {
            track.appendChild(buildTickerItem(entry));
        });
    }

    document.addEventListener("DOMContentLoaded", init);
})();
