.PHONY: install install_all install_modules download_flux download_vae

# install everything except WAN‑VACE downloads
install:
	./install.sh install

# install everything + WAN‑VACE + HF login
install_all:
	./install.sh all

# lower‑level helpers
install_modules:
	./install.sh modules

download_flux:
	./install.sh flux

download_vae:
	./install.sh vae
