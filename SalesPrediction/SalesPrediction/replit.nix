{ pkgs }: {
  deps = [
    pkgs.pango
    pkgs.cairo
    pkgs.gdk-pixbuf
    pkgs.libffi
    pkgs.fontconfig
  ];
}