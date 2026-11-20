() => {
  const visible = e => {
    const s = getComputedStyle(e), r = e.getBoundingClientRect();
    return s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0' && r.width > 0 && r.height > 0;
  };
  const text = (document.body?.innerText || '').trim().replace(/\s+/g, ' ');
  const loadingText = /^(loading|please wait|加载中|正在加载|请稍候|请稍等)[\s.。…!！]*$/i.test(text);
  const controls = [...document.querySelectorAll('button,a[href],input:not([type=hidden]),select,textarea,[role=button]')].some(visible);
  const media = [...document.querySelectorAll('canvas,img,svg,video,iframe,object,embed')]
    .some(e => visible(e) && !e.closest('[role=progressbar]'));
  const loader = [...document.querySelectorAll('[role=progressbar],[aria-busy=true]')].some(visible);
  if (!controls && !media && (loadingText || (!text && loader))) return 'loading_only';
  if (text && !loadingText || controls) return 'content';
  return media ? 'visual_candidate' : 'inspect_pixels';
}
