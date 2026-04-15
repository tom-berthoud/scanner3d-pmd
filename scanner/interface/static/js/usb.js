/* ============================================================
   Scanner 3D — USB Export Logic
   ============================================================ */

window.copyToUsb = async function() {
  try {
    // Fetch available USB drives
    var resp = await fetch('/usb/drives');
    var drives = await resp.json();

    if (!drives || drives.length === 0) {
      showToast('Aucune cle USB detectee. Inserez une cle USB.', 'error');
      return;
    }

    var mountpoint;
    if (drives.length === 1) {
      // Single drive — confirm
      var label = drives[0].label || 'USB';
      var size = drives[0].size || '';
      if (!confirm('Copier le scan sur ' + label + ' (' + size + ') ?')) return;
      mountpoint = drives[0].mountpoint;
    } else {
      // Multiple drives — let user pick
      var msg = 'Plusieurs cles USB detectees :\n';
      for (var i = 0; i < drives.length; i++) {
        msg += (i + 1) + '. ' + (drives[i].label || 'USB') + ' (' + (drives[i].size || '') + ') — ' + drives[i].mountpoint + '\n';
      }
      msg += '\nEntrez le numero (1-' + drives.length + ') :';
      var choice = prompt(msg);
      if (!choice) return;
      var idx = parseInt(choice) - 1;
      if (idx < 0 || idx >= drives.length) {
        showToast('Choix invalide.', 'error');
        return;
      }
      mountpoint = drives[idx].mountpoint;
    }

    // Copy file
    showToast('Copie en cours...', 'info');
    var copyResp = await fetch('/usb/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mountpoint: mountpoint })
    });
    var result = await copyResp.json();

    if (copyResp.ok) {
      showToast('Copie reussie : ' + result.path, 'success');
    } else {
      showToast(result.error || 'Erreur lors de la copie.', 'error');
    }
  } catch (e) {
    showToast('Erreur reseau : ' + e, 'error');
  }
};
