DeLong URL-matching metadata table

Rows: 6533
This is the constructed metadata table for matching local HTML files
to historical URLs. Original archive ZIPs were not modified.

Filled
------
Identity and HTML-extracted fields (title, date, permalink, prev/next,
Movable Type entry id).

Wayback Machine via the public CDX index
  found: 6513
  not_found: 20
  wayback_url is the earliest status-200 capture of the extracted permalink.
  Bodies were not downloaded or text-compared.

Live URL check
  Sample permalinks on j-bradford-delong.net were probed on 2026-11-06.
  That result was applied to other rows on the same host.
  broken/timeout rows: 6533

Intentionally not invented
--------------------------
current_url / current_url_source / current_url_status
  Some posts were later reprinted on Substack or WordPress, but there is
  no 1:1 URL map yet. These stay not_checked rather than guessed.

match_method is url_exact only when the permalink itself appears in CDX.
That is a URL match, not a body-text match.
